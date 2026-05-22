#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Next steps for decode performance:
# [ ] Keep multiple decode length variants loaded at once instead of compiling one static bin per run
# [ ] Opportunity to fuse data layout transformations (e.g., transpose ops) onto end of other operations (e.g., transpose after RoPE)
# [ ] Some kernels are not optimized; e.g., softmax masking is using scalar cores
# [ ] Fine-tune parameters of operators (e.g., num AIE columns, tile sizes)
# [ ] KV cache layout is still group-major; seq-major slots would reduce append sync calls further
# [ ] Spatial fusion of operators

import torch
from pathlib import Path
from models.exported_llama3 import llama_inference_harness as harness
from models.exported_llama3.aie_buffers import AIELlamaBuffers
from models.exported_llama3.aie_operators import AIELlamaOperators
from models.exported_llama3.decode_packet_cache import (
    append_decode_kv_cache,
    initialize_decode_packet_cache,
)
from models.exported_llama3.llama_packed_weights import (
    default_llama_packed_weights_dir,
    write_llama_packed_weight_artifact,
)
from models.exported_llama3.runtime_config import select_compiled_seq_len
import logging


class LlamaNpuRunner:
    def __init__(self, config, static_seq_len):
        self.config = config
        self.max_seq_len = static_seq_len
        self.aie_ops = AIELlamaOperators(config, static_seq_len)
        self.aie_buffers = AIELlamaBuffers(config, static_seq_len, self.aie_ops)

    def forward_pass(self, config, state):
        if config is not self.config:
            raise ValueError("LlamaNpuRunner was called with a different config")
        _, seq_len = state.token_ids.shape
        if seq_len > 1:
            ret = self.prefill(state)
            state.num_preceding_tokens = state.token_ids.shape[1]
            self.sync_prefill_cache_to_decode(state)
            return ret

        ret = self.decode(state)
        state.num_preceding_tokens += 1
        return ret

    def prefill(self, state):
        return _prefill_forward_pass(self, self.config, state)

    def decode(self, state):
        return _decode_forward_pass(self, self.config, state)

    def sync_prefill_cache_to_decode(self, state):
        # Pack prefill KV state into the fused decode packet cache.
        for layer_idx in range(self.config.n_layers):
            initialize_decode_packet_cache(
                self.config,
                self.aie_ops,
                self.max_seq_len,
                layer_idx,
                self.aie_buffers.keys_cache[layer_idx].to_torch(),
                self.aie_buffers.values_cache[layer_idx].to_torch(),
                state.num_preceding_tokens,
            )


# Prefill
# ##########################################################################


def grouped_query_attention_forward_prefill(
    runner,
    config,
    x,
    keys_cache,
    values_cache,
    layer_idx,
    mask=None,
):
    aie_ops = runner.aie_ops
    aie_buffers = runner.aie_buffers
    batch, seq_len, emb_dim = x.shape
    num_preceding_tokens = keys_cache.shape[2]

    # Step 1: Linear projections
    aie_ops.prefill.attn_query(
        aie_buffers.prefill.x_norm,
        aie_buffers.W_attn_query_prefill[layer_idx],
        aie_buffers.prefill.queries,
    )
    aie_ops.prefill.attn_key(
        aie_buffers.prefill.x_norm,
        aie_buffers.W_attn_key_prefill[layer_idx],
        aie_buffers.prefill.keys,
    )
    aie_ops.prefill.attn_value(
        aie_buffers.prefill.x_norm,
        aie_buffers.W_attn_value_prefill[layer_idx],
        aie_buffers.prefill.values,
    )

    # Step 2: Apply RoPE to queries and keys
    aie_ops.prefill.rope_queries(
        aie_buffers.prefill.queries,
        aie_buffers.prefill.rope_angles,
        aie_buffers.prefill.queries,
    )
    aie_ops.prefill.rope_keys(
        aie_buffers.prefill.keys,
        aie_buffers.prefill.rope_angles,
        aie_buffers.prefill.keys,
    )

    # Read results from NPU; to_torch() syncs from device internally
    queries = aie_buffers.prefill.queries.to_torch()[: seq_len * config.n_heads, :]
    keys = aie_buffers.prefill.keys.to_torch()[: seq_len * config.n_kv_groups, :]
    values = aie_buffers.prefill.values.to_torch()[
        :seq_len, :
    ]  # (seq_len, n_kv_groups * head_dim)
    queries = queries.view(batch, seq_len, config.n_heads, config.head_dim)
    keys = keys.unsqueeze(0).view(batch, seq_len, config.n_kv_groups, config.head_dim)
    values = values.unsqueeze(0).view(
        batch, seq_len, config.n_kv_groups, config.head_dim
    )  # (batch, seq_len, num_kv_groups, head_dim)

    # Step 3: Transpose for attention computation
    # As a result of the attention projections, the queries, keys and values for each head are interspersed with each other.
    # Transpose so that heads are consecutive for attention computation:
    # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
    queries = queries.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    keys = keys.transpose(1, 2)  # (batch, num_kv_groups, seq_len, head_dim)
    values = values.transpose(1, 2)  # (batch, num_kv_groups, seq_len, head_dim)

    # Step 4: Combine newly computed keys/values for most recent token with cache; these values are used as the updated cache and will be returned to use in the next iteration.
    keys_cache = torch.cat([keys_cache, keys], dim=2)
    values_cache = torch.cat([values_cache, values], dim=2)
    keys = keys_cache
    values = values_cache

    # Step 5: Repeat keys and values for grouped attention -- multiple queries get the same key/value
    group_size = config.n_heads // config.n_kv_groups
    values = values.repeat_interleave(group_size, dim=1)
    context_len = keys.shape[2]

    # Step 6: Compute attention scores using NPU (per-head)
    # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, context_len)
    # -> (batch, num_heads, seq_len, context_len)

    queries_buf = aie_buffers.prefill.attn_scores_queries_all.torch_view().view(
        config.n_heads, -1, config.head_dim
    )
    queries_buf[:, :seq_len, :] = queries.squeeze(0)[
        :, :seq_len, :
    ]  # (num_heads, seq_len, head_dim)
    keys_buf = aie_buffers.prefill.attn_scores_keys_all.torch_view().view(
        config.n_kv_groups, config.head_dim, -1
    )
    keys_buf[:, :, :context_len] = keys.squeeze(0).transpose(
        -2, -1
    )  # (num_kv_groups, head_dim, context_len)

    # Transfer parent buffers to NPU once
    aie_buffers.prefill.attn_scores_queries_all.to("npu")
    aie_buffers.prefill.attn_scores_keys_all.to("npu")
    aie_buffers.prefill.attn_scores.to("npu")

    # Execute GEMM for each head using sub-buffers
    for h in range(config.n_heads):
        kv_group = h // group_size
        aie_ops.prefill.attn_scores(
            aie_buffers.prefill.attn_scores_queries_per_head[h],
            aie_buffers.prefill.attn_scores_keys_per_kv_group[kv_group],
            aie_buffers.prefill.attn_scores_per_head[h],
        )

    # Read back all results at once from parent buffer and apply scaling on NPU
    aie_ops.prefill.attn_scale(
        aie_buffers.prefill.attn_scores,
        aie_buffers.prefill.attn_scale_factor,
        aie_buffers.prefill.attn_scores,
    )
    # Buffer is (n_heads * max_seq_len, max_seq_len), view as (n_heads, max_seq_len, max_seq_len) then slice
    max_seq_len_buf = aie_buffers.prefill.attn_scores.shape[0] // config.n_heads
    scores = (
        aie_buffers.prefill.attn_scores.to_torch()  # to_torch() syncs device→host; torch_view() would not sync and must not be used here
        .view(config.n_heads, max_seq_len_buf, max_seq_len_buf)
        .unsqueeze(0)[:, :, :seq_len, :context_len]
    )

    # Step 7: Apply mask
    # This ensures causality, so that tokens in the future cannot attend to tokens in the past.
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 8: Apply softmax on CPU
    scores = torch.softmax(scores.to(torch.float32), dim=-1).to(torch.bfloat16)
    attention_weights = scores

    # Step 9: Compute attention output
    # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
    # -> (batch, num_heads, seq_len, head_dim)
    context = torch.matmul(attention_weights, values)

    # Step 10: Concatenate heads and project
    # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads * head_dim)
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    output = torch.nn.functional.linear(
        context, config.weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
    )

    return output, keys_cache, values_cache


def swiglu_ffn_forward_prefill(runner, layer_idx):
    aie_ops = runner.aie_ops
    aie_buffers = runner.aie_buffers

    # Step 1: Gate projection
    aie_ops.prefill.ffn_up_gate(
        aie_buffers.prefill.x_norm,
        aie_buffers.W_ffn_gate_prefill[layer_idx],
        aie_buffers.prefill.ffn_gate,
    )

    # Step 2: Up projection
    aie_ops.prefill.ffn_up_gate(
        aie_buffers.prefill.x_norm,
        aie_buffers.W_ffn_up_prefill[layer_idx],
        aie_buffers.prefill.ffn_up,
    )

    # Step 3: Apply SiLU activation
    aie_ops.prefill.ffn_silu(aie_buffers.prefill.ffn_gate, aie_buffers.prefill.ffn_gate)

    # Step 4: Element-wise multiplication
    aie_ops.prefill.eltwise_mul_ffn(
        aie_buffers.prefill.ffn_gate,
        aie_buffers.prefill.ffn_up,
        aie_buffers.prefill.ffn_hidden,
    )

    # Step 5: Down projection
    aie_ops.prefill.ffn_down(
        aie_buffers.prefill.ffn_hidden,
        aie_buffers.W_ffn_down_prefill[layer_idx],
        aie_buffers.prefill.ffn_output,
    )


def transformer_block_forward_prefill(
    runner,
    config,
    seq_len,
    layer_idx,
    attn_keys_cache,
    attn_values_cache,
    attn_mask,
):
    aie_ops = runner.aie_ops
    aie_buffers = runner.aie_buffers

    # Step 1: RMS normalization
    aie_ops.prefill.rms_norm(
        aie_buffers.prefill.x,
        aie_buffers.W_norm1[layer_idx],
        aie_buffers.prefill.x_norm,
    )
    x_norm = aie_buffers.prefill.x_norm.to_torch().unsqueeze(0)[:, :seq_len, :]

    # Step 2: Attention
    attn_output, attn_keys, attn_values = grouped_query_attention_forward_prefill(
        runner,
        config,
        x_norm,
        attn_keys_cache,
        attn_values_cache,
        layer_idx,
        attn_mask,
    )

    # Step 3: Residual
    aie_buffers.prefill.attn_output.torch_view().unsqueeze(0)[
        0, :seq_len, :
    ] = attn_output
    aie_buffers.prefill.attn_output.to("npu")
    aie_ops.prefill.residual_add(
        aie_buffers.prefill.x, aie_buffers.prefill.attn_output, aie_buffers.prefill.x
    )
    x = aie_buffers.prefill.x.to_torch().unsqueeze(0)[:, :seq_len, :]

    # Step 4: Post-norm
    aie_buffers.prefill.x.torch_view().unsqueeze(0)[0, :seq_len, :] = x
    aie_buffers.prefill.x.to("npu")
    aie_ops.prefill.rms_norm(
        aie_buffers.prefill.x,
        aie_buffers.W_norm2[layer_idx],
        aie_buffers.prefill.x_norm,
    )
    x_norm = aie_buffers.prefill.x_norm.to_torch().unsqueeze(0)[:, :seq_len, :]

    # Step 5: Feed-forward network
    swiglu_ffn_forward_prefill(runner, layer_idx)

    # Step 6: Residual
    aie_ops.prefill.residual_add(
        aie_buffers.prefill.x, aie_buffers.prefill.ffn_output, aie_buffers.prefill.x
    )

    return attn_keys, attn_values


def _prefill_forward_pass(runner, config, state):
    aie_ops = runner.aie_ops
    aie_buffers = runner.aie_buffers

    batch, seq_len = state.token_ids.shape

    # Step 1: RoPE angles
    num_preceding_tokens = state.attn_keys_caches[0].shape[2]
    angles_slice = config.angles[num_preceding_tokens : num_preceding_tokens + seq_len]
    aie_buffers.prefill.rope_angles.torch_view()[:seq_len, :] = angles_slice
    aie_buffers.prefill.rope_angles.to("npu")

    # Step 2: Token embedding
    tok_emb_weight = config.weights["model.embed_tokens.weight"]
    x = torch.nn.functional.embedding(state.token_ids, tok_emb_weight)
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
    )
    aie_buffers.prefill.x.torch_view().unsqueeze(0)[0, :seq_len, :] = x
    aie_buffers.prefill.x.to("npu")

    # Step 3: Transformer blocks
    for layer_idx in range(config.n_layers):
        (
            state.attn_keys_caches[layer_idx],
            state.attn_values_caches[layer_idx],
        ) = transformer_block_forward_prefill(
            runner,
            config,
            seq_len,
            layer_idx,
            state.attn_keys_caches[layer_idx],
            state.attn_values_caches[layer_idx],
            attn_mask=attn_mask,
        )

    # Step 4: Final normalization
    aie_ops.prefill.rms_norm(
        aie_buffers.prefill.x, aie_buffers.W_final_norm, aie_buffers.prefill.x
    )

    # Step 5: Output projection
    for i in range(config.vocab_partitions):
        aie_ops.prefill.out_head(
            aie_buffers.prefill.x,
            aie_buffers.W_out_head_parts[i],
            aie_buffers.prefill.logits_parts[i],
        )
    logits_padded_partitioned = aie_buffers.prefill.logits.to_torch()
    logits_padded = (
        logits_padded_partitioned.transpose(0, 1)
        .contiguous()
        .view(-1, config.padded_vocab_size)
    )
    logits = logits_padded.unsqueeze(0)[:, :seq_len, : config.vocab_size]

    # Step 6: Initialize per-layer NPU cache buffers with current cache state for decode phase
    for layer_idx in range(config.n_layers):
        cache_len = state.attn_keys_caches[layer_idx].shape[2]
        aie_buffers.keys_cache[layer_idx].torch_view()[:, :cache_len, :] = (
            state.attn_keys_caches[layer_idx].squeeze(0)
        )
        aie_buffers.values_cache[layer_idx].torch_view()[:, :cache_len, :] = (
            state.attn_values_caches[layer_idx].squeeze(0)
        )
        aie_buffers.keys_cache[layer_idx].to("npu")
        aie_buffers.values_cache[layer_idx].to("npu")

    return logits, state


# Decode
# ##########################################################################

def _decode_forward_pass(runner, config, state):
    aie_ops = runner.aie_ops
    max_seq_len = runner.max_seq_len

    batch, seq_len = state.token_ids.shape
    assert seq_len == 1
    assert state.num_preceding_tokens < max_seq_len

    # Prefill RoPE angle look-up tables
    angles_slice = config.angles[
        state.num_preceding_tokens : state.num_preceding_tokens + seq_len
    ]
    aie_ops.decode.fused.get_buffer("rope_angles").torch_view()[
        :
    ] = angles_slice.flatten()

    # Token embedding (on CPU)
    tok_emb_weight = config.weights["model.embed_tokens.weight"]
    x = torch.nn.functional.embedding(state.token_ids, tok_emb_weight)
    aie_ops.decode.fused.get_buffer("x").torch_view().view(-1, config.emb_dim)[
        :seq_len, :
    ] = x

    # Fused NPU operator for all of decode (16 transformer blocks + final norm + final linear layer)
    aie_ops.decode.fused.input_buffer.to("cpu")
    aie_ops.decode.fused()  # FusedFullELFCallable.__call__() syncs output_buffer to cpu
    append_decode_kv_cache(config, aie_ops, max_seq_len, state.num_preceding_tokens)
    logits = aie_ops.decode.fused.get_buffer("logits").torch_view().view(
        1, 1, config.vocab_size
    )

    return logits, state


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = harness.parse_args()

    required_seq_len = args.prompt_len + args.num_tokens
    max_seq_len = select_compiled_seq_len(required_seq_len)
    logging.info(
        "Using static sequence length %d for %d requested positions",
        max_seq_len,
        required_seq_len,
    )

    prompt = harness.get_prompt(args.prompt_len)

    config, state = harness.init(args.weights_path, args.tokenizer_path, prompt=prompt)
    packed_weights_dir = (
        Path(args.packed_weights_dir)
        if args.packed_weights_dir is not None
        else default_llama_packed_weights_dir(args.weights_path)
    )
    config.packed_weights_dir = packed_weights_dir
    config.require_packed_weights = args.require_packed_weights

    if args.prepare_weights:
        manifest = write_llama_packed_weight_artifact(config, packed_weights_dir)
        print(f"packed_weights_dir: {packed_weights_dir}")
        print(f"packed_weights_file: {packed_weights_dir / 'weights.bf16.bin'}")
        print(f"packed_manifest_file: {packed_weights_dir / 'manifest.json'}")
        print(f"packed_total_bytes: {manifest['total_bytes']}")
        return

    runner = LlamaNpuRunner(config, max_seq_len)

    print(prompt, end="", flush=True)
    harness.generate(
        config,
        state,
        runner.forward_pass,
        use_kv_cache=True,
        num_tokens=args.num_tokens,
    )


if __name__ == "__main__":
    main()
