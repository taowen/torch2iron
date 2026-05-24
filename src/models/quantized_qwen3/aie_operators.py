#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from iron.common.context import AIEContext
from torch2iron.fusion import FusedFullELFCallable, FusedMLIROperator

from models.quantized_qwen3.generated.decode_fused import build_decode_fused_op
from models.quantized_qwen3.generated.prefill_layout import PREFILL_LAYER_WEIGHT_SPECS
from models.quantized_qwen3.generated.prefill_operators import build_prefill_fused_op
from models.quantized_qwen3.operators.w4a16_gemv.op import W4A16GEMV
from models.quantized_qwen3.qwen_weight_layout import iter_qwen_decode_weight_specs
from models.quantized_qwen3.runtime_config import (
    DECODE_ATTN_CHUNK_SIZE,
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
        self.prefill.fused.qparam_buffer.to("npu")
        self.prefill.fused.scratch_buffer.to("npu")
        self.prefill.fused.output_buffer.to("npu")
        self._build_prefill_lm_head_op(config, prefill_build_suffix)

    def _build_prefill_lm_head_op(self, config, build_suffix):
        context = AIEContext(build_dir=Path("build_elf") / f"{build_suffix}_lm_head")
        op = W4A16GEMV(
            M=config.vocab_size,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=8,
            tile_size_output=16,
            context=context,
        )
        fused_op = FusedMLIROperator(
            "prefill_lm_head_w4a16",
            [(op, "W_out_head_qparam", "x", "logits")],
            input_args=["x"],
            output_args=["logits"],
            external_args={"qparam": ["W_out_head_qparam"]},
            compile_mode="full_elf_dynamic",
            context=context,
        ).compile()
        fused = fused_op.get_callable()

        _, qparam = config.weight_store.linear_qparam("lm_head")
        fused.mark_buffer_dirty("qparam")
        fused.get_buffer("W_out_head_qparam").torch_view()[:] = qparam.flatten()
        fused.qparam_buffer.to("npu")
        fused.input_buffer.to("npu")
        fused.output_buffer.to("npu")

        self.prefill.lm_head_fused_op = fused_op
        self.prefill.lm_head_fused = fused

    def _build_decode_ops(self, config, prompt_len, build_suffix, *, prefill_seq_len):
        self.decode.variant_seq_lens = iter_decode_variant_seq_lens(prompt_len)
        self.decode.variants = {}

        shared_weight_buffer = None
        shared_qparam_buffer = None

        for variant_seq_len in self.decode.variant_seq_lens:
            variant_suffix = f"decode{variant_seq_len}_chunk{DECODE_ATTN_CHUNK_SIZE}"
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
                fused.qparam_buffer.to("npu")
                shared_weight_buffer = fused.weight_buffer
                shared_qparam_buffer = fused.qparam_buffer
            else:
                fused.replace_buffer("weight", shared_weight_buffer)
                fused.replace_buffer("qparam", shared_qparam_buffer)

            fused.input_buffer.to("npu")
            fused.scratch_buffer.to("npu")
            fused.output_buffer.to("npu")
            self.decode.variants[variant_seq_len] = AIEDecodeVariant(
                max_seq_len=variant_seq_len,
                fused_op=fused_op,
                fused=fused,
                current_cache_slot=current_cache_slot,
            )

def _is_linear_source(source: str) -> bool:
    return source.endswith(
        (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
            "lm_head.weight",
        )
    )


def _linear_prefix(source: str) -> str:
    if source == "lm_head.weight":
        return "lm_head"
    if not source.endswith(".weight"):
        raise ValueError(f"not a linear weight source: {source}")
    return source[: -len(".weight")]


def load_decode_weight_buffers(config, fused):
    logging.info("Loading decode W4A16 weights from packed artifact: %s", config.packed_weights_dir)
    fused.mark_buffer_dirty("weight")
    fused.mark_buffer_dirty("qparam")

    for spec in iter_qwen_decode_weight_specs(config):
        if _is_linear_source(spec["source"]):
            _linear_spec, qparam = config.weight_store.linear_qparam(
                _linear_prefix(spec["source"])
            )
            fused.get_buffer(f"{spec['name']}_qparam").torch_view()[:] = qparam.flatten()
        else:
            fused.get_buffer(spec["name"]).torch_view()[:] = config.weights[spec["source"]].flatten()


def _copy_buffer(fused, name, tensor):
    fused.get_buffer(name).torch_view()[:] = tensor.flatten()


def load_prefill_fused_weight_buffers(config, fused):
    fused.mark_buffer_dirty("weight")
    fused.mark_buffer_dirty("qparam")
    for layer_idx in range(config.n_layers):
        prefix = f"model.layers.{layer_idx}"
        for name, source_suffix, transpose in PREFILL_LAYER_WEIGHT_SPECS:
            source = f"{prefix}.{source_suffix}"
            target_name = f"{name}_{layer_idx}"
            if _is_linear_source(source):
                _linear_spec, gemm_weight = config.weight_store.linear_gemm_weight(
                    _linear_prefix(source)
                )
                fused.get_buffer(f"{target_name}_qparam").torch_view()[:] = gemm_weight.flatten()
            else:
                _copy_buffer(fused, target_name, config.weights[source])

    _copy_buffer(fused, "W_final_norm", config.weights["model.norm.weight"])
