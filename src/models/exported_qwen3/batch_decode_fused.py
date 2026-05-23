#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-batch Qwen3 decode fused operator.

This is intentionally separate from the generated single-request decode path.
It keeps the current single-request implementation stable while proving the
batch decode shape needed to improve NPU utilization.
"""

from __future__ import annotations

from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator
from torch2iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMM,
    LlamaChunkedAttention,
    RMSNorm,
    RoPE,
    SiLU,
    StridedCopy,
)

from models.exported_qwen3.generated.decode_layout import (
    DECODE_TRANSFORMER_WEIGHT_NAMES,
    EXPECTED_DECODE_LAYERS,
)
from models.exported_qwen3.runtime_config import DECODE_ATTN_CHUNK_SIZE


BF16_BYTES = 2
BATCH_DECODE_ROWS = 32
BATCH_DECODE_COLUMNS = 8
BATCH_DECODE_GEMM_TILE_M = 8
BATCH_DECODE_GEMM_TILE_K = 64
BATCH_DECODE_GEMM_TILE_N = 64


def _bytes(elements: int) -> int:
    return elements * BF16_BYTES


def _slice(name: str, start_elements: int, length_elements: int) -> str:
    start = start_elements * BF16_BYTES
    end = (start_elements + length_elements) * BF16_BYTES
    return f"{name}[{start}:{end}]"


def _batch_packet_cache_names(config, batch_size: int) -> list[str]:
    return [
        f"packet_cache_{layer_idx}_{batch_idx}"
        for layer_idx in range(config.n_layers)
        for batch_idx in range(batch_size)
    ]


def _present_key_name(layer_idx: int, batch_idx: int) -> str:
    return f"present_keys_{layer_idx}_{batch_idx}"


def _present_value_name(layer_idx: int, batch_idx: int) -> str:
    return f"present_values_{layer_idx}_{batch_idx}"


def _gemm(config, context, *, k: int, n: int, tile_n: int | None = None) -> GEMM:
    return GEMM(
        M=BATCH_DECODE_ROWS,
        K=k,
        N=n,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        tile_m=BATCH_DECODE_GEMM_TILE_M,
        tile_k=BATCH_DECODE_GEMM_TILE_K,
        tile_n=tile_n or BATCH_DECODE_GEMM_TILE_N,
        b_col_maj=False,
        separate_c_tiles=False,
        emulate_bf16_mmul_with_bfp16=False,
        context=context,
    )


def _strided_copy(
    context,
    *,
    input_sizes,
    input_strides,
    input_offset=0,
    output_sizes,
    output_strides,
    output_offset=0,
    input_buffer_size,
    output_buffer_size,
    num_aie_channels=1,
):
    return StridedCopy(
        input_sizes=input_sizes,
        input_strides=input_strides,
        input_offset=input_offset,
        output_sizes=output_sizes,
        output_strides=output_strides,
        output_offset=output_offset,
        input_buffer_size=input_buffer_size,
        output_buffer_size=output_buffer_size,
        num_aie_channels=num_aie_channels,
        context=context,
    )


def build_batch_decode_fused_op(config, max_seq_len, batch_size, build_suffix):
    if config.n_layers != EXPECTED_DECODE_LAYERS:
        raise ValueError(
            f"batch decode expects {EXPECTED_DECODE_LAYERS} layers, got {config.n_layers}"
        )
    if batch_size <= 1:
        raise ValueError("batch decode requires batch_size > 1")
    if batch_size > BATCH_DECODE_ROWS:
        raise ValueError(
            f"batch_size {batch_size} exceeds padded rows {BATCH_DECODE_ROWS}"
        )
    if max_seq_len % DECODE_ATTN_CHUNK_SIZE != 0:
        raise ValueError(
            f"max_seq_len must be divisible by {DECODE_ATTN_CHUNK_SIZE}"
        )

    context = AIEContext(build_dir=Path("build_batch_elf") / build_suffix)

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    n_kv_groups = config.n_kv_groups
    q_heads_per_group = n_heads // n_kv_groups
    head_dim = config.head_dim
    attn_dim = n_heads * head_dim
    kv_dim = n_kv_groups * head_dim

    x_elements = BATCH_DECODE_ROWS * emb_dim
    q_elements = BATCH_DECODE_ROWS * attn_dim
    kv_elements = BATCH_DECODE_ROWS * kv_dim
    ffn_elements = BATCH_DECODE_ROWS * hidden_dim
    packet_chunk_elements = 2 * DECODE_ATTN_CHUNK_SIZE * head_dim + DECODE_ATTN_CHUNK_SIZE
    packet_elements_per_group = (
        max_seq_len // DECODE_ATTN_CHUNK_SIZE * packet_chunk_elements
    )
    packet_elements = n_kv_groups * packet_elements_per_group

    current_cache_slot = max_seq_len - 1
    current_chunk = current_cache_slot // DECODE_ATTN_CHUNK_SIZE
    current_row = current_cache_slot % DECODE_ATTN_CHUNK_SIZE
    current_k_packet_offset = (
        current_chunk * packet_chunk_elements + current_row * head_dim
    )
    current_v_packet_offset = (
        current_chunk * packet_chunk_elements
        + DECODE_ATTN_CHUNK_SIZE * head_dim
        + current_row * head_dim
    )

    rms_norm_op = RMSNorm(
        size=x_elements,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_channels=1,
        tile_size=emb_dim,
        weighted=True,
        context=context,
    )
    attn_query_norm_op = RMSNorm(
        size=q_elements,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_channels=1,
        tile_size=head_dim,
        weighted=True,
        context=context,
    )
    attn_key_norm_op = RMSNorm(
        size=kv_elements,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_channels=1,
        tile_size=head_dim,
        weighted=True,
        context=context,
    )
    rope_queries_op = RoPE(
        rows=BATCH_DECODE_ROWS * n_heads,
        cols=head_dim,
        angle_rows=BATCH_DECODE_ROWS,
        context=context,
    )
    rope_keys_op = RoPE(
        rows=BATCH_DECODE_ROWS * n_kv_groups,
        cols=head_dim,
        angle_rows=BATCH_DECODE_ROWS,
        context=context,
    )
    gemm_attn_query_op = _gemm(config, context, k=emb_dim, n=attn_dim)
    gemm_attn_key_value_op = _gemm(config, context, k=emb_dim, n=kv_dim)
    gemm_attn_output_op = _gemm(config, context, k=attn_dim, n=emb_dim)
    gemm_ffn_up_gate_op = _gemm(config, context, k=emb_dim, n=hidden_dim)
    gemm_ffn_down_op = _gemm(config, context, k=hidden_dim, n=emb_dim)
    silu_ffn_op = SiLU(
        size=ffn_elements,
        tile_size=hidden_dim // BATCH_DECODE_COLUMNS,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        context=context,
    )
    eltwise_mul_ffn_op = ElementwiseMul(
        size=ffn_elements,
        tile_size=hidden_dim // BATCH_DECODE_COLUMNS,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        context=context,
    )
    residual_add_op = ElementwiseAdd(
        size=x_elements,
        tile_size=emb_dim // BATCH_DECODE_COLUMNS,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        context=context,
    )
    copy_present_kv_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, head_dim),
        input_strides=(head_dim, 1),
        output_sizes=(n_kv_groups, head_dim),
        output_strides=(head_dim, 1),
        input_buffer_size=kv_dim,
        output_buffer_size=kv_dim,
    )
    strided_copy_packet_key_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, head_dim),
        input_strides=(head_dim, 1),
        output_sizes=(n_kv_groups, head_dim),
        output_strides=(packet_elements_per_group, 1),
        output_offset=current_k_packet_offset,
        input_buffer_size=kv_dim,
        output_buffer_size=packet_elements,
    )
    strided_copy_packet_value_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, head_dim),
        input_strides=(head_dim, 1),
        output_sizes=(n_kv_groups, head_dim),
        output_strides=(packet_elements_per_group, 1),
        output_offset=current_v_packet_offset,
        input_buffer_size=kv_dim,
        output_buffer_size=packet_elements,
    )
    llama_chunked_attention_op = LlamaChunkedAttention(
        max_seq_len=max_seq_len,
        num_kv_groups=n_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=DECODE_ATTN_CHUNK_SIZE,
        context=context,
    )

    runlist = []
    for layer_idx in range(config.n_layers):
        runlist.extend(
            [
                (rms_norm_op, "x", f"W_norm1_{layer_idx}", "x_norm"),
                (
                    gemm_attn_query_op,
                    "x_norm",
                    f"W_attn_query_{layer_idx}",
                    "queries",
                ),
                (
                    gemm_attn_key_value_op,
                    "x_norm",
                    f"W_attn_key_{layer_idx}",
                    "keys",
                ),
                (
                    gemm_attn_key_value_op,
                    "x_norm",
                    f"W_attn_value_{layer_idx}",
                    "values",
                ),
                (
                    attn_query_norm_op,
                    "queries",
                    f"W_attn_query_norm_{layer_idx}",
                    "queries",
                ),
                (
                    attn_key_norm_op,
                    "keys",
                    f"W_attn_key_norm_{layer_idx}",
                    "keys",
                ),
                (rope_queries_op, "queries", "rope_angles", "queries"),
                (rope_keys_op, "keys", "rope_angles", "keys"),
            ]
        )
        for batch_idx in range(batch_size):
            q_slice = _slice("queries", batch_idx * attn_dim, attn_dim)
            k_slice = _slice("keys", batch_idx * kv_dim, kv_dim)
            v_slice = _slice("values", batch_idx * kv_dim, kv_dim)
            context_slice = _slice("attn_context", batch_idx * attn_dim, attn_dim)
            packet_name = f"packet_cache_{layer_idx}_{batch_idx}"
            runlist.extend(
                [
                    (
                        copy_present_kv_op,
                        k_slice,
                        _present_key_name(layer_idx, batch_idx),
                    ),
                    (
                        copy_present_kv_op,
                        v_slice,
                        _present_value_name(layer_idx, batch_idx),
                    ),
                    (strided_copy_packet_key_op, k_slice, packet_name),
                    (strided_copy_packet_value_op, v_slice, packet_name),
                    (
                        llama_chunked_attention_op,
                        q_slice,
                        packet_name,
                        context_slice,
                    ),
                ]
            )
        runlist.extend(
            [
                (
                    gemm_attn_output_op,
                    "attn_context",
                    f"W_attn_output_decode_{layer_idx}",
                    "attn_output",
                ),
                (residual_add_op, "x", "attn_output", "x"),
                (rms_norm_op, "x", f"W_norm2_{layer_idx}", "x_norm"),
                (
                    gemm_ffn_up_gate_op,
                    "x_norm",
                    f"W_ffn_gate_{layer_idx}",
                    "ffn_gate",
                ),
                (
                    gemm_ffn_up_gate_op,
                    "x_norm",
                    f"W_ffn_up_{layer_idx}",
                    "ffn_up",
                ),
                (silu_ffn_op, "ffn_gate", "ffn_gate"),
                (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
                (
                    gemm_ffn_down_op,
                    "ffn_hidden",
                    f"W_ffn_down_{layer_idx}",
                    "ffn_output",
                ),
                (residual_add_op, "x", "ffn_output", "x"),
            ]
        )

    runlist.append((rms_norm_op, "x", "W_final_norm", "hidden_out"))

    output_args = [
        "hidden_out",
        *[
            _present_key_name(layer_idx, batch_idx)
            for layer_idx in range(config.n_layers)
            for batch_idx in range(batch_size)
        ],
        *[
            _present_value_name(layer_idx, batch_idx)
            for layer_idx in range(config.n_layers)
            for batch_idx in range(batch_size)
        ],
    ]

    return (
        FusedMLIROperator(
            "batch_decode_fused_op",
            runlist,
            input_args=["x", "rope_angles"],
            output_args=output_args,
            buffer_sizes={
                "x": _bytes(x_elements),
                "x_norm": _bytes(x_elements),
                "queries": _bytes(q_elements),
                "keys": _bytes(kv_elements),
                "values": _bytes(kv_elements),
                "attn_context": _bytes(q_elements),
                "attn_output": _bytes(x_elements),
                "ffn_gate": _bytes(ffn_elements),
                "ffn_up": _bytes(ffn_elements),
                "ffn_hidden": _bytes(ffn_elements),
                "ffn_output": _bytes(x_elements),
                "hidden_out": _bytes(x_elements),
                **{
                    name: _bytes(packet_elements)
                    for name in _batch_packet_cache_names(config, batch_size)
                },
            },
            external_args={
                "weight": list(DECODE_TRANSFORMER_WEIGHT_NAMES),
                "kv_cache": _batch_packet_cache_names(config, batch_size),
            },
            compile_mode="full_elf_dynamic",
            context=context,
        ).compile(),
        current_cache_slot,
    )


def batch_packet_cache_names(config, batch_size: int) -> list[str]:
    return _batch_packet_cache_names(config, batch_size)


def present_key_name(layer_idx: int, batch_idx: int) -> str:
    return _present_key_name(layer_idx, batch_idx)


def present_value_name(layer_idx: int, batch_idx: int) -> str:
    return _present_value_name(layer_idx, batch_idx)
