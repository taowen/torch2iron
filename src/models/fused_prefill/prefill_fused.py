#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written fused prefill prototype.

The upstream IRON Llama path only fuses decode. Prefill is still a Python
sequence of AIE dispatches with CPU attention in the middle. This module starts
the replacement path: build one FusedMLIROperator for the full static prefill
graph, including per-layer full-sequence attention.

This is intentionally hand-owned in ``models.fused_prefill`` and is not emitted
by ``codegen.py``.
"""

from __future__ import annotations

from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator
from torch2iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMM,
    RMSNorm,
    RoPE,
    SiLU,
    Softmax,
    StridedCopy,
)


BF16_BYTES = 2
PREFILL_NUM_AIE_COLUMNS = 8
PREFILL_GEMM_TILE_SIZE = 64
PREFILL_VOCAB_PARTITIONS = 4


def _bytes(elements: int) -> int:
    return elements * BF16_BYTES


def _slice(name: str, start_elements: int, length_elements: int) -> str:
    start = _bytes(start_elements)
    end = _bytes(start_elements + length_elements)
    return f"{name}[{start}:{end}]"


def _configure_prefill_vocab(config) -> None:
    min_n = (
        PREFILL_GEMM_TILE_SIZE
        * PREFILL_NUM_AIE_COLUMNS
        * PREFILL_VOCAB_PARTITIONS
    )
    config.padded_vocab_size = (config.vocab_size + min_n - 1) // min_n * min_n
    config.vocab_partitions = PREFILL_VOCAB_PARTITIONS


def _prefill_gemm(
    config,
    prompt_len,
    context,
    *,
    k,
    n,
    num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
    b_col_maj=False,
    separate_c_tiles=False,
):
    return GEMM(
        M=prompt_len,
        K=k,
        N=n,
        num_aie_columns=num_aie_columns,
        tile_m=PREFILL_GEMM_TILE_SIZE,
        tile_k=PREFILL_GEMM_TILE_SIZE,
        tile_n=PREFILL_GEMM_TILE_SIZE,
        b_col_maj=b_col_maj,
        separate_c_tiles=separate_c_tiles,
        context=context,
    )


def _strided_copy(
    context,
    *,
    input_sizes,
    input_strides,
    output_sizes,
    output_strides,
    input_buffer_size,
    output_buffer_size,
    num_aie_channels=1,
):
    return StridedCopy(
        input_sizes=input_sizes,
        input_strides=input_strides,
        input_offset=0,
        output_sizes=output_sizes,
        output_strides=output_strides,
        output_offset=0,
        input_buffer_size=input_buffer_size,
        output_buffer_size=output_buffer_size,
        num_aie_channels=num_aie_channels,
        context=context,
    )


def _lm_head_part_name(part_idx: int) -> str:
    return f"W_out_head_part_{part_idx}"


def _layer_weight_names(config) -> list[str]:
    names = []
    for layer_idx in range(config.n_layers):
        names.extend(
            [
                f"W_norm1_{layer_idx}",
                f"W_attn_query_{layer_idx}",
                f"W_attn_key_{layer_idx}",
                f"W_attn_value_{layer_idx}",
                f"W_attn_output_{layer_idx}",
                f"W_norm2_{layer_idx}",
                f"W_ffn_gate_{layer_idx}",
                f"W_ffn_up_{layer_idx}",
                f"W_ffn_down_{layer_idx}",
            ]
        )
    names.append("W_final_norm")
    return names


def _present_key_name(layer_idx: int) -> str:
    return f"present_keys_{layer_idx}"


def _present_value_name(layer_idx: int) -> str:
    return f"present_values_{layer_idx}"


def build_prefill_fused_op(config, prompt_len, build_suffix):
    _configure_prefill_vocab(config)

    context = AIEContext(build_dir=Path("build_prefill_elf") / build_suffix)
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    n_kv_groups = config.n_kv_groups
    q_heads_per_group = n_heads // n_kv_groups
    head_dim = config.head_dim

    x_elements = prompt_len * emb_dim
    q_elements = prompt_len * n_heads * head_dim
    kv_elements = prompt_len * n_kv_groups * head_dim
    scores_elements = n_heads * prompt_len * prompt_len
    head_q_elements = prompt_len * head_dim
    head_score_elements = prompt_len * prompt_len
    kv_group_elements = prompt_len * head_dim
    logits_elements = prompt_len * config.padded_vocab_size
    logits_part_width = config.padded_vocab_size // config.vocab_partitions
    logits_part_elements = prompt_len * logits_part_width

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
        config, prompt_len, context, k=emb_dim, n=n_heads * head_dim
    )
    attn_key_value_op = _prefill_gemm(
        config, prompt_len, context, k=emb_dim, n=n_kv_groups * head_dim
    )
    rope_queries_op = RoPE(
        rows=prompt_len * n_heads,
        cols=head_dim,
        angle_rows=prompt_len,
        context=context,
    )
    rope_keys_op = RoPE(
        rows=prompt_len * n_kv_groups,
        cols=head_dim,
        angle_rows=prompt_len,
        context=context,
    )

    query_layout_op = _strided_copy(
        context,
        input_sizes=(n_heads, prompt_len, head_dim),
        input_strides=(head_dim, n_heads * head_dim, 1),
        output_sizes=(n_heads, prompt_len, head_dim),
        output_strides=(prompt_len * head_dim, head_dim, 1),
        input_buffer_size=q_elements,
        output_buffer_size=q_elements,
        num_aie_channels=1,
    )
    key_scores_layout_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, head_dim, prompt_len),
        input_strides=(head_dim, 1, n_kv_groups * head_dim),
        output_sizes=(n_kv_groups, head_dim, prompt_len),
        output_strides=(head_dim * prompt_len, prompt_len, 1),
        input_buffer_size=kv_elements,
        output_buffer_size=kv_elements,
        num_aie_channels=1,
    )
    key_cache_layout_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, prompt_len, head_dim),
        input_strides=(head_dim, n_kv_groups * head_dim, 1),
        output_sizes=(n_kv_groups, prompt_len, head_dim),
        output_strides=(prompt_len * head_dim, head_dim, 1),
        input_buffer_size=kv_elements,
        output_buffer_size=kv_elements,
        num_aie_channels=1,
    )
    value_cache_layout_op = key_cache_layout_op
    value_repeat_layout_op = _strided_copy(
        context,
        input_sizes=(n_kv_groups, q_heads_per_group, prompt_len, head_dim),
        input_strides=(head_dim, 0, n_kv_groups * head_dim, 1),
        output_sizes=(n_kv_groups, q_heads_per_group, prompt_len, head_dim),
        output_strides=(
            q_heads_per_group * prompt_len * head_dim,
            prompt_len * head_dim,
            head_dim,
            1,
        ),
        input_buffer_size=kv_elements,
        output_buffer_size=q_elements,
        num_aie_channels=1,
    )

    attn_scores_op = _prefill_gemm(
        config, prompt_len, context, k=head_dim, n=prompt_len
    )
    attn_scale_op = ElementwiseMul(
        size=scores_elements,
        tile_size=prompt_len,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        context=context,
    )
    attn_mask_add_op = ElementwiseAdd(
        size=scores_elements,
        tile_size=prompt_len,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        context=context,
    )
    softmax_op = Softmax(
        rows=n_heads * prompt_len,
        cols=prompt_len,
        num_aie_columns=1,
        num_channels=1,
        rtp_vector_size=prompt_len,
        context=context,
    )
    attn_context_op = _prefill_gemm(
        config,
        prompt_len,
        context,
        k=prompt_len,
        n=head_dim,
        num_aie_columns=1,
    )
    context_layout_op = _strided_copy(
        context,
        input_sizes=(prompt_len, n_heads, head_dim),
        input_strides=(head_dim, prompt_len * head_dim, 1),
        output_sizes=(prompt_len, n_heads, head_dim),
        output_strides=(n_heads * head_dim, head_dim, 1),
        input_buffer_size=q_elements,
        output_buffer_size=q_elements,
        num_aie_channels=1,
    )
    attn_output_op = _prefill_gemm(config, prompt_len, context, k=emb_dim, n=emb_dim)

    ffn_up_gate_op = _prefill_gemm(config, prompt_len, context, k=emb_dim, n=hidden_dim)
    ffn_silu_op = SiLU(
        size=prompt_len * hidden_dim,
        tile_size=hidden_dim,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        context=context,
    )
    ffn_mul_op = ElementwiseMul(
        size=prompt_len * hidden_dim,
        tile_size=hidden_dim,
        num_aie_columns=PREFILL_NUM_AIE_COLUMNS,
        context=context,
    )
    ffn_down_op = _prefill_gemm(config, prompt_len, context, k=hidden_dim, n=emb_dim)
    lm_head_op = _prefill_gemm(
        config,
        prompt_len,
        context,
        k=emb_dim,
        n=logits_part_width,
        b_col_maj=True,
        separate_c_tiles=True,
    )

    runlist = []
    for layer_idx in range(config.n_layers):
        runlist.extend(
            [
                (rms_norm_op, "x", f"W_norm1_{layer_idx}", "x_norm"),
                (attn_query_op, "x_norm", f"W_attn_query_{layer_idx}", "queries_raw"),
                (attn_key_value_op, "x_norm", f"W_attn_key_{layer_idx}", "keys_raw"),
                (
                    attn_key_value_op,
                    "x_norm",
                    f"W_attn_value_{layer_idx}",
                    "values_raw",
                ),
                (rope_queries_op, "queries_raw", "rope_angles", "queries_raw"),
                (rope_keys_op, "keys_raw", "rope_angles", "keys_raw"),
                (query_layout_op, "queries_raw", "queries"),
                (key_scores_layout_op, "keys_raw", "keys_for_scores"),
                (key_cache_layout_op, "keys_raw", _present_key_name(layer_idx)),
                (value_cache_layout_op, "values_raw", _present_value_name(layer_idx)),
                (value_repeat_layout_op, "values_raw", "values"),
            ]
        )

        for head_idx in range(n_heads):
            kv_group = head_idx // q_heads_per_group
            runlist.append(
                (
                    attn_scores_op,
                    _slice("queries", head_idx * head_q_elements, head_q_elements),
                    _slice(
                        "keys_for_scores",
                        kv_group * head_q_elements,
                        head_q_elements,
                    ),
                    _slice(
                        "attn_scores",
                        head_idx * head_score_elements,
                        head_score_elements,
                    ),
                )
            )

        runlist.extend(
            [
                (attn_scale_op, "attn_scores", "attn_scale_factor", "attn_scores"),
                (attn_mask_add_op, "attn_scores", "attn_mask", "attn_scores"),
                (softmax_op, "attn_scores", "attn_weights"),
            ]
        )

        for head_idx in range(n_heads):
            kv_group = head_idx // q_heads_per_group
            runlist.append(
                (
                    attn_context_op,
                    _slice(
                        "attn_weights",
                        head_idx * head_score_elements,
                        head_score_elements,
                    ),
                    _slice("values", kv_group * head_q_elements, head_q_elements),
                    _slice(
                        "attn_context_heads",
                        head_idx * head_q_elements,
                        head_q_elements,
                    ),
                )
            )

        runlist.extend(
            [
                (context_layout_op, "attn_context_heads", "attn_context"),
                (
                    attn_output_op,
                    "attn_context",
                    f"W_attn_output_{layer_idx}",
                    "attn_output",
                ),
                (residual_add_op, "x", "attn_output", "x"),
                (rms_norm_op, "x", f"W_norm2_{layer_idx}", "x_norm"),
                (ffn_up_gate_op, "x_norm", f"W_ffn_gate_{layer_idx}", "ffn_gate"),
                (ffn_up_gate_op, "x_norm", f"W_ffn_up_{layer_idx}", "ffn_up"),
                (ffn_silu_op, "ffn_gate", "ffn_gate"),
                (ffn_mul_op, "ffn_gate", "ffn_up", "ffn_hidden"),
                (ffn_down_op, "ffn_hidden", f"W_ffn_down_{layer_idx}", "ffn_output"),
                (residual_add_op, "x", "ffn_output", "x"),
            ]
        )

    runlist.append((rms_norm_op, "x", "W_final_norm", "hidden_out"))
    for part_idx in range(config.vocab_partitions):
        runlist.append(
            (
                lm_head_op,
                "hidden_out",
                _lm_head_part_name(part_idx),
                _slice(
                    "logits",
                    part_idx * logits_part_elements,
                    logits_part_elements,
                ),
            )
        )

    output_args = [
        "logits",
        *[_present_key_name(i) for i in range(config.n_layers)],
        *[_present_value_name(i) for i in range(config.n_layers)],
    ]

    return FusedMLIROperator(
        "prefill_fused_op",
        runlist,
        input_args=["x", "rope_angles", "attn_mask"],
        output_args=output_args,
        buffer_sizes={
            "logits": _bytes(logits_elements),
            **{
                _present_key_name(i): _bytes(kv_elements)
                for i in range(config.n_layers)
            },
            **{
                _present_value_name(i): _bytes(kv_elements)
                for i in range(config.n_layers)
            },
        },
        external_args={
            "weight": _layer_weight_names(config),
            "lm_head": [
                _lm_head_part_name(i) for i in range(config.vocab_partitions)
            ],
        },
        context=context,
    ).compile()
