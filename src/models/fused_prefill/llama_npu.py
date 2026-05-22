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
from models.fused_prefill import llama_inference_harness as harness
from models.fused_prefill.aie_buffers import AIELlamaBuffers
from models.fused_prefill.aie_operators import AIELlamaOperators
from models.fused_prefill.decode_packet_cache import (
    append_decode_kv_cache,
    decode_packet_slot_offsets,
    sync_decode_packet_range,
)
from models.fused_prefill.generated.decode_layout import DECODE_PACKET_CACHE_NAMES
from models.fused_prefill.llama_packed_weights import (
    default_llama_packed_weights_dir,
    write_llama_packed_weight_artifact,
)
from models.fused_prefill.fused_prefill_runtime import fused_prefill_forward_pass
from models.fused_prefill.runtime_config import select_compiled_seq_len
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
        return fused_prefill_forward_pass(self, state)

    def decode(self, state):
        return _decode_forward_pass(self, self.config, state)

    def sync_prefill_cache_to_decode(self, state):
        # Prefill and decode share the packet cache BO. Mark decode's fixed
        # scratch slot as valid for the current token; no full cache copy is
        # needed at the prefill/decode boundary.
        current_slot = self.aie_ops.decode.current_cache_slot
        for layer_idx in range(self.config.n_layers):
            packet_cache = self.aie_ops.decode.fused.get_buffer(
                DECODE_PACKET_CACHE_NAMES[layer_idx]
            )
            packet = packet_cache.torch_view()
            for group_idx in range(self.config.n_kv_groups):
                _, _, mask_offset = decode_packet_slot_offsets(
                    self.config,
                    self.max_seq_len,
                    group_idx,
                    current_slot,
                )
                packet[mask_offset] = 1.0
                sync_decode_packet_range(packet_cache, mask_offset, 1)


# Decode
# ##########################################################################


def _decode_forward_pass(runner, config, state):
    aie_ops = runner.aie_ops
    max_seq_len = runner.max_seq_len

    _, seq_len = state.token_ids.shape
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
