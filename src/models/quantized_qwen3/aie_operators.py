#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from types import SimpleNamespace

from iron.common.context import AIEContext
from torch2iron.fusion import load_elf

from models.quantized_qwen3.batch_decode_fused import (
    batch_decode_dense_weight_names,
    batch_decode_qparam_names,
    batch_packet_cache_names,
    build_batch_decode_fused_op,
    select_batch_decode_rows,
)
from models.quantized_qwen3.generated.prefill_layout import PREFILL_LAYER_WEIGHT_SPECS
from models.quantized_qwen3.generated.prefill_operators import build_prefill_fused_op
from models.quantized_qwen3.qwen_weight_layout import iter_qwen_decode_weight_specs
from models.quantized_qwen3.runtime_config import (
    DECODE_ATTN_CHUNK_SIZE,
    select_prefill_chunk_config,
)


class AIEQwenOperators:
    def __init__(self, config, prefill_seq_len, decode_max_seq_len, batch_size):
        build_suffix = (
            f"prefill{prefill_seq_len}_decode{decode_max_seq_len}_batch{batch_size}"
        )
        self.context = AIEContext(build_dir=Path("build") / build_suffix)
        self.context.build_dir.mkdir(parents=True, exist_ok=True)

        self.prefill = SimpleNamespace()
        self.decode = SimpleNamespace()

        self._build_prefill_ops(config, prefill_seq_len)
        self._build_decode_ops(
            config,
            decode_max_seq_len,
            batch_size=batch_size,
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
        self.prefill.body_fused_op = build_prefill_fused_op(
            config,
            prompt_len,
            prefill_build_suffix,
            chunk_size=prefill_config.chunk_size,
            compute_rows=prefill_config.compute_rows,
            q_head_block_size=prefill_config.q_head_block_size,
        )
        final_prefill_build_suffix = f"{prefill_build_suffix}_final_lm_head"
        self.prefill.final_fused_op = build_prefill_fused_op(
            config,
            prompt_len,
            final_prefill_build_suffix,
            chunk_size=prefill_config.chunk_size,
            compute_rows=prefill_config.compute_rows,
            q_head_block_size=prefill_config.q_head_block_size,
            include_lm_head=True,
        )
        self.prefill.body_elf_data = load_elf(self.prefill.body_fused_op)
        self.prefill.final_elf_data = load_elf(self.prefill.final_fused_op)
        self.prefill.fused_op = self.prefill.final_fused_op
        self.prefill.fused = self.prefill.final_fused_op.get_callable()
        self.prefill.loaded_elf_kind = "final"

        load_prefill_fused_weight_buffers(config, self.prefill.fused)
        self.prefill.fused.weight_buffer.to("npu")
        self.prefill.fused.qparam_buffer.to("npu")
        self.prefill.fused.scratch_buffer.to("npu")
        self.prefill.fused.output_buffer.to("npu")

    def _build_decode_ops(self, config, max_seq_len, *, batch_size):
        decode_suffix = (
            f"batch{batch_size}_rows{select_batch_decode_rows(batch_size)}"
            f"_decode{max_seq_len}_chunk{DECODE_ATTN_CHUNK_SIZE}_xfm"
        )
        fused_op, current_cache_slot, decode_rows = build_batch_decode_fused_op(
            config,
            max_seq_len,
            batch_size,
            decode_suffix,
        )
        fused = fused_op.get_callable()
        load_batch_decode_weight_buffers(config, fused)
        fused.weight_buffer.to("npu")
        fused.qparam_buffer.to("npu")
        fused.input_buffer.to("npu")
        fused.scratch_buffer.to("npu")
        fused.output_buffer.to("npu")
        for name in batch_packet_cache_names(config, batch_size):
            packet_cache = fused.get_buffer(name)
            packet_cache.torch_view().zero_()
            packet_cache.to("npu")

        self.decode.max_seq_len = max_seq_len
        self.decode.batch_size = batch_size
        self.decode.fused_op = fused_op
        self.decode.fused = fused
        self.decode.current_cache_slot = current_cache_slot
        self.decode.decode_rows = decode_rows


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


def load_batch_decode_weight_buffers(config, fused):
    logging.info(
        "Loading batch decode W4A16 GEMM weights from packed artifact: %s",
        config.packed_weights_dir,
    )
    fused.mark_buffer_dirty("weight")
    fused.mark_buffer_dirty("qparam")

    specs = {spec["name"]: spec for spec in iter_qwen_decode_weight_specs(config)}
    for name in batch_decode_dense_weight_names():
        source = specs[name]["source"]
        fused.get_buffer(name).torch_view()[:] = config.weights[source].flatten()

    qparam_names = set(batch_decode_qparam_names())
    for spec in specs.values():
        qparam_name = f"{spec['name']}_qparam"
        if qparam_name not in qparam_names:
            continue
        _linear_spec, w4_weight = config.weight_store.linear_gemm_w4_weight(
            _linear_prefix(spec["source"])
        )
        fused.get_buffer(qparam_name).torch_view()[:] = w4_weight.flatten()

    for layer_idx in range(config.n_layers):
        prefix = f"model.layers.{layer_idx}"
        _pair_spec, kv_weight = config.weight_store.linear_paired_gemm_w4_weight(
            f"{prefix}.self_attn.kv_proj"
        )
        fused.get_buffer(f"W_attn_key_value_{layer_idx}_qparam").torch_view()[
            :
        ] = kv_weight.flatten()
        _pair_spec, gate_up_weight = config.weight_store.linear_paired_gemm_w4_weight(
            f"{prefix}.mlp.gate_up_proj"
        )
        fused.get_buffer(f"W_ffn_gate_up_{layer_idx}_qparam").torch_view()[
            :
        ] = gate_up_weight.flatten()


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
                _linear_spec, w4_weight = config.weight_store.linear_gemm_w4_weight(
                    _linear_prefix(source)
                )
                fused.get_buffer(f"{target_name}_qparam").torch_view()[:] = w4_weight.flatten()
            else:
                _copy_buffer(fused, target_name, config.weights[source])

    _copy_buffer(fused, "W_final_norm", config.weights["model.norm.weight"])
    _linear_spec, lm_head_w4_weight = config.weight_store.linear_gemm_w4_weight("lm_head")
    fused.get_buffer("W_out_head_qparam").torch_view()[:] = lm_head_w4_weight.flatten()
