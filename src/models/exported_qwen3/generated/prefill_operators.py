#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated fused prefill ELF builders for exported_qwen3.

Regenerate with:
    uv run python -m torch2iron.export.codegen --model-package models.exported_qwen3

The model topology, layer count, and layer weight list come from
torch.export.ExportedProgram. Runtime-specific tiling parameters still enter
through ``build_prefill_fused_op`` because they are optimization choices rather
than model semantics.
"""

from __future__ import annotations

from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator
from torch2iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMM,
    GEMV,
    LlamaChunkedPrefillAttention,
    RMSNorm,
    RoPE,
    SiLU,
    StridedCopy,
)

from models.exported_qwen3.generated.decode_layout import DECODE_PACKET_CACHE_NAMES
from models.exported_qwen3.generated.prefill_layout import (
    EXPECTED_PREFILL_LAYERS,
    PREFILL_LAYER_WEIGHT_SPECS,
)
from models.exported_qwen3.runtime_config import DECODE_ATTN_CHUNK_SIZE


BF16_BYTES = 2
PREFILL_NUM_AIE_COLUMNS = 8
PREFILL_GEMM_TILE_SIZE = 64
PREFILL_CHUNK_GEMM_TILE_M = 8
PREFILL_VOCAB_PARTITIONS = 4


def _bytes(elements: int) -> int:
    return elements * BF16_BYTES


def _configure_prefill_vocab(config) -> None:
    min_n = (
        PREFILL_GEMM_TILE_SIZE
        * PREFILL_NUM_AIE_COLUMNS
        * PREFILL_VOCAB_PARTITIONS
    )
    config.padded_vocab_size = (config.vocab_size + min_n - 1) // min_n * min_n
    config.vocab_partitions = PREFILL_VOCAB_PARTITIONS


def _prefill_lm_head_tile_size_output(config) -> int:
    rows_per_column = config.vocab_size // PREFILL_NUM_AIE_COLUMNS
    for tile_size_output in (32, 16, 8, 4):
        if rows_per_column % tile_size_output == 0:
            return tile_size_output
    raise ValueError(
        "prefill lm_head vocab rows per AIE column must be divisible by 4, 8, 16, or 32"
    )


def _prefill_gemm(
    config,
    query_len,
    context,
    *,
    k,
    n,
    b_col_maj=False,
    separate_c_tiles=False,
):
    return GEMM(
        M=query_len,
        K=k,
        N=n,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        tile_m=PREFILL_CHUNK_GEMM_TILE_M,
        tile_k=PREFILL_GEMM_TILE_SIZE,
        tile_n=PREFILL_GEMM_TILE_SIZE,
        b_col_maj=b_col_maj,
        separate_c_tiles=separate_c_tiles,
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


def _layer_weight_names(config) -> list[str]:
    names = []
    for layer_idx in range(config.n_layers):
        names.extend(
            f"{name}_{layer_idx}"
            for name, _source_suffix, _transpose in PREFILL_LAYER_WEIGHT_SPECS
        )
    names.append("W_final_norm")
    return names


def _present_key_name(layer_idx: int) -> str:
    return f"present_keys_{layer_idx}"


def _present_value_name(layer_idx: int) -> str:
    return f"present_values_{layer_idx}"


def build_prefill_fused_op(
    config,
    max_seq_len,
    build_suffix,
    *,
    chunk_size,
    compute_rows,
    q_head_block_size,
    dry_run=False,
):
    _configure_prefill_vocab(config)
    if config.n_layers != EXPECTED_PREFILL_LAYERS:
        raise ValueError(
            f"chunked prefill expects {EXPECTED_PREFILL_LAYERS} layers, "
            f"got {config.n_layers}"
        )
    context = AIEContext(build_dir=Path("build_prefill_elf") / build_suffix)
    query_len = chunk_size
    if compute_rows < query_len:
        raise ValueError("PREFILL_CHUNK_COMPUTE_ROWS must cover PREFILL_CHUNK_SIZE")
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    n_kv_groups = config.n_kv_groups
    q_heads_per_group = n_heads // n_kv_groups
    attn_q_head_block_size = min(q_head_block_size, q_heads_per_group)
    if q_heads_per_group % attn_q_head_block_size != 0:
        raise ValueError("q_heads_per_group must be divisible by q_head_block_size")
    q_head_blocks_per_group = q_heads_per_group // attn_q_head_block_size
    head_dim = config.head_dim
    kv_dim = n_kv_groups * head_dim

    x_elements = compute_rows * emb_dim
    q_elements = compute_rows * n_heads * head_dim
    kv_elements = compute_rows * kv_dim
    kv_attn_elements = query_len * kv_dim
    q_elements_per_attn_group = query_len * attn_q_head_block_size * head_dim
    current_kv_elements_per_group = query_len * head_dim
    q_current_elements_per_group = (
        q_elements_per_attn_group + 2 * current_kv_elements_per_group
    )
    q_current_elements = n_kv_groups * q_current_elements_per_group
    attn_grouped_elements = n_kv_groups * q_elements_per_attn_group
    packet_chunk_elements = (
        2 * DECODE_ATTN_CHUNK_SIZE * head_dim + DECODE_ATTN_CHUNK_SIZE
    )
    packet_elements_per_group = (
        max_seq_len // DECODE_ATTN_CHUNK_SIZE * packet_chunk_elements
    )
    packet_elements = n_kv_groups * packet_elements_per_group

    rms_norm_op = RMSNorm(
        size=x_elements,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        num_channels=1,
        tile_size=emb_dim,
        weighted=True,
        context=context,
    )
    residual_add_op = ElementwiseAdd(
        size=x_elements,
        tile_size=emb_dim,
        context=context,
    )
    attn_query_op = _prefill_gemm(
        config, compute_rows, context, k=emb_dim, n=n_heads * head_dim
    )
    attn_key_value_op = _prefill_gemm(
        config, compute_rows, context, k=emb_dim, n=kv_dim
    )
    rope_queries_op = RoPE(
        rows=compute_rows * n_heads,
        cols=head_dim,
        angle_rows=compute_rows,
        context=context,
    )
    rope_keys_op = RoPE(
        rows=compute_rows * n_kv_groups,
        cols=head_dim,
        angle_rows=compute_rows,
        context=context,
    )
    attn_query_norm_op = RMSNorm(
        size=q_elements,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        num_channels=1,
        tile_size=head_dim,
        weighted=True,
        context=context,
    )
    attn_key_norm_op = RMSNorm(
        size=kv_elements,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        num_channels=1,
        tile_size=head_dim,
        weighted=True,
        context=context,
    )
    pack_queries_ops = [
        _strided_copy(
            context,
            input_sizes=(
                n_kv_groups,
                query_len,
                attn_q_head_block_size,
                head_dim,
            ),
            input_strides=(
                q_heads_per_group * head_dim,
                n_heads * head_dim,
                head_dim,
                1,
            ),
            input_offset=q_block_idx * attn_q_head_block_size * head_dim,
            output_sizes=(
                n_kv_groups,
                query_len,
                attn_q_head_block_size,
                head_dim,
            ),
            output_strides=(
                q_current_elements_per_group,
                attn_q_head_block_size * head_dim,
                head_dim,
                1,
            ),
            input_buffer_size=q_elements,
            output_buffer_size=q_current_elements,
        )
        for q_block_idx in range(q_head_blocks_per_group)
    ]
    pack_keys_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, query_len, head_dim),
        input_strides=(head_dim, kv_dim, 1),
        output_sizes=(n_kv_groups, query_len, head_dim),
        output_strides=(
            q_current_elements_per_group,
            head_dim,
            1,
        ),
        output_offset=q_elements_per_attn_group,
        input_buffer_size=kv_elements,
        output_buffer_size=q_current_elements,
    )
    pack_values_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, query_len, head_dim),
        input_strides=(head_dim, kv_dim, 1),
        output_sizes=(n_kv_groups, query_len, head_dim),
        output_strides=(
            q_current_elements_per_group,
            head_dim,
            1,
        ),
        output_offset=q_elements_per_attn_group + current_kv_elements_per_group,
        input_buffer_size=kv_elements,
        output_buffer_size=q_current_elements,
    )
    present_kv_copy_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, query_len, head_dim),
        input_strides=(head_dim, kv_dim, 1),
        output_sizes=(n_kv_groups, query_len, head_dim),
        output_strides=(query_len * head_dim, head_dim, 1),
        input_buffer_size=kv_elements,
        output_buffer_size=kv_attn_elements,
    )
    llama_chunked_prefill_attention_op = LlamaChunkedPrefillAttention(
        max_seq_len=max_seq_len,
        query_len=query_len,
        num_kv_groups=n_kv_groups,
        q_heads_per_group=attn_q_head_block_size,
        q_head_block_size=attn_q_head_block_size,
        head_dim=head_dim,
        chunk_size=DECODE_ATTN_CHUNK_SIZE,
        context=context,
    )
    unpack_context_ops = [
        _strided_copy(
            context,
            input_sizes=(
                n_kv_groups,
                query_len,
                attn_q_head_block_size,
                head_dim,
            ),
            input_strides=(
                q_elements_per_attn_group,
                attn_q_head_block_size * head_dim,
                head_dim,
                1,
            ),
            output_sizes=(
                n_kv_groups,
                query_len,
                attn_q_head_block_size,
                head_dim,
            ),
            output_strides=(
                q_heads_per_group * head_dim,
                n_heads * head_dim,
                head_dim,
                1,
            ),
            output_offset=q_block_idx * attn_q_head_block_size * head_dim,
            input_buffer_size=attn_grouped_elements,
            output_buffer_size=q_elements,
        )
        for q_block_idx in range(q_head_blocks_per_group)
    ]
    attn_output_op = _prefill_gemm(
        config, compute_rows, context, k=n_heads * head_dim, n=emb_dim
    )
    ffn_up_gate_op = _prefill_gemm(config, compute_rows, context, k=emb_dim, n=hidden_dim)
    ffn_silu_op = SiLU(
        size=compute_rows * hidden_dim,
        tile_size=hidden_dim,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        context=context,
    )
    ffn_mul_op = ElementwiseMul(
        size=compute_rows * hidden_dim,
        tile_size=hidden_dim,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        context=context,
    )
    ffn_down_op = _prefill_gemm(config, compute_rows, context, k=hidden_dim, n=emb_dim)

    runlist = []
    for layer_idx in range(config.n_layers):
        runlist.append((rms_norm_op, "x", f"W_norm1_{layer_idx}", "x_norm"))
        runlist.append(
            (
                attn_query_op,
                "x_norm",
                f"W_attn_query_prefill_{layer_idx}",
                "queries",
            )
        )
        runlist.append(
            (
                attn_key_value_op,
                "x_norm",
                f"W_attn_key_prefill_{layer_idx}",
                "keys",
            )
        )
        runlist.append(
            (
                attn_key_value_op,
                "x_norm",
                f"W_attn_value_prefill_{layer_idx}",
                "values",
            )
        )
        runlist.append(
            (
                attn_query_norm_op,
                "queries",
                f"W_attn_query_norm_{layer_idx}",
                "queries",
            )
        )
        runlist.append(
            (
                attn_key_norm_op,
                "keys",
                f"W_attn_key_norm_{layer_idx}",
                "keys",
            )
        )
        runlist.append((rope_queries_op, "queries", "rope_angles", "queries"))
        runlist.extend(
            [
                (rope_keys_op, "keys", "rope_angles", "keys"),
                (present_kv_copy_op, "keys", _present_key_name(layer_idx)),
                (present_kv_copy_op, "values", _present_value_name(layer_idx)),
            ]
        )
        for q_block_idx in range(q_head_blocks_per_group):
            runlist.extend(
                [
                    (pack_queries_ops[q_block_idx], "queries", "qkv_current"),
                    (pack_keys_op, "keys", "qkv_current"),
                    (pack_values_op, "values", "qkv_current"),
                    (
                        llama_chunked_prefill_attention_op,
                        "qkv_current",
                        DECODE_PACKET_CACHE_NAMES[layer_idx],
                        "attn_context_grouped",
                    ),
                    (
                        unpack_context_ops[q_block_idx],
                        "attn_context_grouped",
                        "attn_context",
                    ),
                ]
            )
        runlist.append(
            (
                attn_output_op,
                "attn_context",
                f"W_attn_output_prefill_{layer_idx}",
                "attn_output",
            )
        )
        runlist.append((residual_add_op, "x", "attn_output", "x"))
        runlist.append((rms_norm_op, "x", f"W_norm2_{layer_idx}", "x_norm"))
        runlist.append(
            (ffn_up_gate_op, "x_norm", f"W_ffn_gate_prefill_{layer_idx}", "ffn_gate")
        )
        runlist.append(
            (ffn_up_gate_op, "x_norm", f"W_ffn_up_prefill_{layer_idx}", "ffn_up")
        )
        runlist.extend(
            [
                (ffn_silu_op, "ffn_gate", "ffn_gate"),
                (ffn_mul_op, "ffn_gate", "ffn_up", "ffn_hidden"),
            ]
        )
        runlist.append(
            (
                ffn_down_op,
                "ffn_hidden",
                f"W_ffn_down_prefill_{layer_idx}",
                "ffn_output",
            )
        )
        runlist.append((residual_add_op, "x", "ffn_output", "x"))

    runlist.append((rms_norm_op, "x", "W_final_norm", "hidden_out"))

    output_args = [
        "hidden_out",
        *[_present_key_name(i) for i in range(config.n_layers)],
        *[_present_value_name(i) for i in range(config.n_layers)],
    ]

    return FusedMLIROperator(
        "prefill_chunk_fused_op",
        runlist,
        input_args=["x", "rope_angles"],
        output_args=output_args,
        buffer_sizes={
            "qkv_current": _bytes(q_current_elements),
            "hidden_out": _bytes(x_elements),
            **{
                name: _bytes(packet_elements)
                for name in DECODE_PACKET_CACHE_NAMES
            },
        },
        external_args={
            "weight": _layer_weight_names(config),
            "kv_cache": list(DECODE_PACKET_CACHE_NAMES),
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile(dry_run=dry_run)


def build_prefill_lm_head_fused_op(config, build_suffix, dry_run=False):
    context = AIEContext(build_dir=Path("build_prefill_elf") / f"{build_suffix}_lm_head")
    lm_head_op = GEMV(
        M=config.vocab_size,
        K=config.emb_dim,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        tile_size_input=4,
        tile_size_output=_prefill_lm_head_tile_size_output(config),
        context=context,
    )
    return FusedMLIROperator(
        "prefill_lm_head_fused_op",
        [(lm_head_op, "W_out_head", "x", "logits")],
        input_args=["x"],
        output_args=["logits"],
        external_args={"lm_head": ["W_out_head"]},
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile(dry_run=dry_run)
