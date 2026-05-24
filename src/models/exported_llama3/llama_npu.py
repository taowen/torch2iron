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
from models.exported_llama3.aie_operators import AIELlamaOperators
from models.exported_llama3.decode_packet_cache import (
    append_decode_kv_cache,
    copy_decode_packet_cache_tokens,
    mark_decode_current_cache_slot,
)
from models.exported_llama3.llama_packed_weights import (
    default_llama_packed_weights_dir,
    write_llama_packed_weight_artifact,
)
from models.exported_llama3.prefill_runtime import prefill_forward_pass
from models.exported_llama3.runtime_config import (
    select_compiled_seq_len,
    select_decode_context_len,
    select_decode_variant_seq_len,
)
import logging


class LlamaNpuRunner:
    def __init__(self, config, prefill_seq_len, decode_max_seq_len):
        self.config = config
        self.prefill_max_seq_len = prefill_seq_len
        self.max_seq_len = decode_max_seq_len
        self.aie_ops = AIELlamaOperators(config, prefill_seq_len, decode_max_seq_len)
        self.active_decode_variant = None

    def forward_pass(self, config, state):
        if config is not self.config:
            raise ValueError("LlamaNpuRunner was called with a different config")
        _, seq_len = state.token_ids.shape
        if seq_len > 1:
            ret = self.prefill(state)
            state.num_preceding_tokens = state.token_ids.shape[1]
            self.ensure_decode_variant(state.num_preceding_tokens)
            return ret

        ret = self.decode(state)
        state.num_preceding_tokens += 1
        return ret

    def prefill(self, state):
        return prefill_forward_pass(self, state)

    def decode(self, state):
        return _decode_forward_pass(self, self.config, state)

    def _select_decode_variant_seq_len(self, valid_tokens_after_decode):
        variant_seq_lens = self.aie_ops.decode.variant_seq_lens
        if valid_tokens_after_decode + 1 <= self.max_seq_len:
            return select_decode_variant_seq_len(
                valid_tokens_after_decode + 1,
                self.max_seq_len,
                variant_seq_lens,
            )
        return select_decode_variant_seq_len(
            valid_tokens_after_decode,
            self.max_seq_len,
            variant_seq_lens,
        )

    def ensure_decode_variant(self, num_valid_tokens):
        target_seq_len = self._select_decode_variant_seq_len(num_valid_tokens + 1)
        target = self.aie_ops.decode.variants[target_seq_len]
        if self.active_decode_variant is target:
            return target

        source = self.active_decode_variant
        if source is None:
            src_fused = self.aie_ops.prefill.fused
            src_max_seq_len = self.prefill_max_seq_len
            sync_src_from_npu = False
        else:
            src_fused = source.fused
            src_max_seq_len = source.max_seq_len
            sync_src_from_npu = True

        if target.fused.kv_cache_buffer is src_fused.kv_cache_buffer:
            mark_decode_current_cache_slot(
                self.config,
                target.fused,
                target.max_seq_len,
                target.current_cache_slot,
            )
        else:
            copy_decode_packet_cache_tokens(
                self.config,
                src_fused,
                src_max_seq_len,
                target.fused,
                target.max_seq_len,
                num_valid_tokens,
                target.current_cache_slot,
                sync_src_from_npu=sync_src_from_npu,
            )

        self.active_decode_variant = target
        logging.info("Using decode seq%d ELF", target.max_seq_len)
        return target


# Decode
# ##########################################################################


def _decode_forward_pass(runner, config, state):
    _, seq_len = state.token_ids.shape
    assert seq_len == 1
    assert state.num_preceding_tokens < runner.max_seq_len
    variant = runner.ensure_decode_variant(state.num_preceding_tokens)
    assert state.num_preceding_tokens < variant.max_seq_len
    fused = variant.fused

    # Prefill RoPE angle look-up tables
    fused.mark_buffer_dirty("input")
    angles_slice = config.angles[
        state.num_preceding_tokens : state.num_preceding_tokens + seq_len
    ]
    fused.get_buffer("rope_angles").torch_view()[:] = angles_slice.flatten()

    # Token embedding (on CPU)
    tok_emb_weight = config.weights["model.embed_tokens.weight"]
    x = torch.nn.functional.embedding(state.token_ids, tok_emb_weight)
    fused.get_buffer("x").torch_view().view(-1, config.emb_dim)[:seq_len, :] = x

    # Fused NPU operator for all of decode (16 transformer blocks + final norm + final linear layer)
    fused()  # FusedFullELFCallable.__call__() syncs output_buffer to cpu
    append_decode_kv_cache(
        config,
        fused,
        variant.max_seq_len,
        variant.current_cache_slot,
        state.num_preceding_tokens,
    )
    logits = fused.get_buffer("logits").torch_view().view(1, 1, config.vocab_size)

    return logits, state


def main():
    logging.basicConfig(level=logging.INFO)
    args = harness.parse_args()

    prompt = harness.get_prompt(args.prompt_len)

    config, state = harness.init(args.weights_path, args.tokenizer_path, prompt=prompt)
    prompt_tokens = state.token_ids.shape[1]
    required_seq_len = prompt_tokens + args.num_tokens
    prefill_seq_len = select_compiled_seq_len(prompt_tokens)
    max_seq_len = select_decode_context_len(required_seq_len)
    logging.info(
        "Using prefill sequence length %d and decode context %d for %d requested positions (%d prompt tokens)",
        prefill_seq_len,
        max_seq_len,
        required_seq_len,
        prompt_tokens,
    )

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

    runner = LlamaNpuRunner(config, prefill_seq_len, max_seq_len)

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
