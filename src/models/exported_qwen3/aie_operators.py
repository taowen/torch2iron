#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from iron.common.context import AIEContext
from torch2iron.fusion import FusedFullELFCallable

from models.exported_qwen3.generated.decode_fused import build_decode_fused_op
from models.exported_qwen3.generated.prefill_layout import PREFILL_LAYER_WEIGHT_SPECS
from models.exported_qwen3.generated.prefill_operators import build_prefill_fused_op
from models.exported_qwen3.qwen_packed_weights import (
    load_qwen_packed_segment,
    validate_qwen_packed_weight_artifact,
)
from models.exported_qwen3.qwen_weight_layout import iter_qwen_decode_weight_specs
from models.exported_qwen3.runtime_config import (
    iter_decode_variant_seq_lens,
    select_prefill_chunk_config,
)


@dataclass
class AIEDecodeVariant:
    max_seq_len: int
    fused_op: object
    fused: FusedFullELFCallable
    current_cache_slot: int


class AIEQwenOperators:
    def __init__(self, config, prefill_seq_len, decode_max_seq_len):
        build_suffix = f"prefill{prefill_seq_len}_decode{decode_max_seq_len}"
        self.context = AIEContext(build_dir=Path("build") / build_suffix)
        self.context.build_dir.mkdir(parents=True, exist_ok=True)

        self.prefill = SimpleNamespace()
        self.decode = SimpleNamespace()

        self._build_prefill_ops(config, prefill_seq_len)
        self._build_decode_ops(
            config,
            decode_max_seq_len,
            build_suffix,
            prefill_seq_len=prefill_seq_len,
        )

    def _build_prefill_ops(self, config, prompt_len):
        prefill_config = select_prefill_chunk_config(prompt_len)
        self.prefill.chunk_size = prefill_config.chunk_size
        self.prefill.compute_rows = prefill_config.compute_rows
        self.prefill.q_head_block_size = prefill_config.q_head_block_size
        prefill_build_suffix = (
            f"seq{prompt_len}_chunk{prefill_config.chunk_size}"
            f"_rows{prefill_config.compute_rows}"
            f"_qhblk{prefill_config.q_head_block_size}"
        )
        self.prefill.fused_op = build_prefill_fused_op(
            config,
            prompt_len,
            prefill_build_suffix,
            chunk_size=prefill_config.chunk_size,
            compute_rows=prefill_config.compute_rows,
            q_head_block_size=prefill_config.q_head_block_size,
        )
        self.prefill.fused = self.prefill.fused_op.get_callable()

        load_prefill_fused_weight_buffers(config, self.prefill.fused)
        self.prefill.fused.weight_buffer.to("npu")
        self.prefill.fused.scratch_buffer.to("npu")
        self.prefill.fused.output_buffer.to("npu")

    def _build_decode_ops(self, config, prompt_len, build_suffix, *, prefill_seq_len):
        self.decode.variant_seq_lens = iter_decode_variant_seq_lens(prompt_len)
        self.decode.variants = {}

        shared_weight_buffer = None
        shared_lm_head_buffer = None

        for variant_seq_len in self.decode.variant_seq_lens:
            variant_suffix = f"decode{variant_seq_len}"
            fused_op, current_cache_slot = build_decode_fused_op(
                config, variant_seq_len, variant_suffix
            )
            fused = FusedFullELFCallable(fused_op)
            if variant_seq_len == prefill_seq_len:
                fused.replace_buffer(
                    "kv_cache",
                    self.prefill.fused.kv_cache_buffer,
                )

            if shared_weight_buffer is None:
                load_decode_weight_buffers(config, fused)
                fused.weight_buffer.to("npu")
                fused.lm_head_buffer.to("npu")
                shared_weight_buffer = fused.weight_buffer
                shared_lm_head_buffer = fused.lm_head_buffer
            else:
                fused.replace_buffer("weight", shared_weight_buffer)
                fused.replace_buffer("lm_head", shared_lm_head_buffer)

            fused.input_buffer.to("npu")
            fused.scratch_buffer.to("npu")
            fused.output_buffer.to("npu")
            self.decode.variants[variant_seq_len] = AIEDecodeVariant(
                max_seq_len=variant_seq_len,
                fused_op=fused_op,
                fused=fused,
                current_cache_slot=current_cache_slot,
            )

def load_decode_weight_buffers(config, fused):
    packed_dir = getattr(config, "packed_weights_dir", None)
    require_packed = getattr(config, "require_packed_weights", False)
    manifest = None
    if packed_dir is not None and Path(packed_dir).exists():
        manifest = validate_qwen_packed_weight_artifact(config, packed_dir)
        logging.info("Loading decode weights from packed artifact: %s", packed_dir)
    elif require_packed:
        raise FileNotFoundError(f"packed weights required but not found at {packed_dir}")
    else:
        logging.info("Loading decode weights from safetensors tensors")

    for spec in iter_qwen_decode_weight_specs(config):
        view = fused.get_buffer(spec["name"]).torch_view()
        if manifest is None:
            view[:] = config.weights[spec["source"]].flatten()
        else:
            view[:] = load_qwen_packed_segment(
                packed_dir, manifest, spec["name"]
            ).flatten()


def _copy_buffer(fused, name, tensor):
    fused.get_buffer(name).torch_view()[:] = tensor.flatten()


def _copy_transposed_buffer(fused, name, tensor):
    fused.get_buffer(name).torch_view()[:] = tensor.T.contiguous().flatten()


def load_prefill_fused_weight_buffers(config, fused):
    for layer_idx in range(config.n_layers):
        prefix = f"model.layers.{layer_idx}"
        for name, source_suffix, transpose in PREFILL_LAYER_WEIGHT_SPECS:
            copy_fn = _copy_transposed_buffer if transpose else _copy_buffer
            copy_fn(
                fused,
                f"{name}_{layer_idx}",
                config.weights[f"{prefix}.{source_suffix}"],
            )

    _copy_buffer(fused, "W_final_norm", config.weights["model.norm.weight"])
