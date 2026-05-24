#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bucketed-row quantized Qwen3 transformer decode fused operator.

Single-request decode is batch decode with one active row.  The transformer
projections, FFN projections, and final norm live in the same ELF.  The AIE
GEMM operators consume compressed W4 tiles from the packed artifact and perform
tile-local dequantization inside the `aie::mmul` kernel.
"""

from __future__ import annotations

import os
from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator
from torch2iron.operators import (
    CopyPresentPacketKV,
    LlamaChunkedAttention,
    RMSNorm,
    RMSNormRoPE,
    ResidualAddRMSNorm,
    SiLUMul,
)

from models.quantized_qwen3.generated.decode_layout import (
    DECODE_LM_HEAD_WEIGHT_NAMES,
    DECODE_TRANSFORMER_WEIGHT_NAMES,
    DECODE_WEIGHT_SPECS,
    EXPECTED_DECODE_LAYERS,
)
from models.quantized_qwen3.operators.w4a16_gemm.op import (
    W4A16GEMM,
    W4A16KGroupGEMM,
    W4A16NShardGEMM,
    W4A16PairedKGroupGEMM,
)
from models.quantized_qwen3.runtime_config import DECODE_ATTN_CHUNK_SIZE


BF16_BYTES = 2
MAX_BATCH_DECODE_ROWS = 32
BATCH_DECODE_ROW_BUCKETS = (4, 8, 16, 32)
BATCH_DECODE_COLUMNS = 8
BATCH_DECODE_GEMM_TILE_M = 8
BATCH_DECODE_GEMM_TILE_K = 128
BATCH_DECODE_TRANSFORMER_TILE_N = 64
BATCH_DECODE_LM_HEAD_TILE_N = 64


def _bytes(elements: int) -> int:
    return elements * BF16_BYTES


def _slice(name: str, start_elements: int, length_elements: int) -> str:
    start = start_elements * BF16_BYTES
    end = (start_elements + length_elements) * BF16_BYTES
    return f"{name}[{start}:{end}]"


def select_batch_decode_rows(batch_size: int) -> int:
    for rows in BATCH_DECODE_ROW_BUCKETS:
        if batch_size <= rows:
            return rows
    raise ValueError(
        f"batch_size {batch_size} exceeds padded rows {MAX_BATCH_DECODE_ROWS}"
    )


def _gemm_tile_m(rows: int) -> int:
    if rows <= 8:
        return 4
    return BATCH_DECODE_GEMM_TILE_M


def _trace_kwargs(build_suffix: str) -> dict:
    trace_size = int(os.environ.get("TORCH2IRON_TRACE_SIZE", "0"))
    if trace_size <= 0:
        return {}
    trace_dir = Path(os.environ.get("TORCH2IRON_TRACE_DIR", "build_trace"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_op_index = os.environ.get("TORCH2IRON_TRACE_OP_INDEX")
    trace_ddr_id = os.environ.get("TORCH2IRON_TRACE_DDR_ID")
    op_index = int(trace_op_index) if trace_op_index is not None else 0
    ddr_id = int(trace_ddr_id) if trace_ddr_id is not None else None
    trace_arg = ddr_id if ddr_id is not None else 3
    trace_suffix = f"{build_suffix}.op{op_index}.arg{trace_arg}.size{trace_size}"
    return {
        "trace_size": trace_size,
        "trace_file": trace_dir / f"{trace_suffix}.trace.txt",
        "trace_json_file": trace_dir / f"{trace_suffix}.trace.json",
        "trace_op_index": op_index,
        "trace_ddr_id": ddr_id,
    }


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


def _spec_by_name() -> dict[str, dict[str, object]]:
    return {spec["name"]: dict(spec) for spec in DECODE_WEIGHT_SPECS}


def _weight_arg(name: str, specs: dict[str, dict[str, object]]) -> str:
    source = str(specs[name]["source"])
    return f"{name}_qparam" if _is_linear_source(source) else name


def batch_decode_dense_weight_names() -> list[str]:
    specs = _spec_by_name()
    return [
        name
        for name in DECODE_TRANSFORMER_WEIGHT_NAMES
        if not _is_linear_source(str(specs[name]["source"]))
    ]


def batch_decode_qparam_names() -> list[str]:
    specs = _spec_by_name()
    names = [
        f"{name}_qparam"
        for name in (*DECODE_TRANSFORMER_WEIGHT_NAMES, *DECODE_LM_HEAD_WEIGHT_NAMES)
        if _is_linear_source(str(specs[name]["source"]))
        and not str(specs[name]["source"]).endswith(
            (
                "k_proj.weight",
                "v_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
            )
        )
    ]
    for layer_idx in range(EXPECTED_DECODE_LAYERS):
        names.append(f"W_attn_key_value_{layer_idx}_qparam")
        names.append(f"W_ffn_gate_up_{layer_idx}_qparam")
    return names


def batch_packet_cache_names(config, batch_size: int) -> list[str]:
    return [
        f"packet_cache_{layer_idx}_{batch_idx}"
        for layer_idx in range(config.n_layers)
        for batch_idx in range(batch_size)
    ]


def present_key_name(layer_idx: int, batch_idx: int) -> str:
    return f"present_keys_{layer_idx}_{batch_idx}"


def present_value_name(layer_idx: int, batch_idx: int) -> str:
    return f"present_values_{layer_idx}_{batch_idx}"


def _gemm(
    context,
    *,
    rows: int,
    k: int,
    n: int,
    tile_n: int = BATCH_DECODE_TRANSFORMER_TILE_N,
) -> W4A16GEMM:
    tile_m = _gemm_tile_m(rows)
    return W4A16GEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_aie_rows=rows // tile_m,
        tile_m=tile_m,
        tile_k=BATCH_DECODE_GEMM_TILE_K,
        tile_n=tile_n,
        context=context,
    )


def _paired_k_group_gemm(
    context,
    *,
    rows: int,
    k: int,
    n: int,
    tile_n: int = BATCH_DECODE_TRANSFORMER_TILE_N,
) -> W4A16PairedKGroupGEMM:
    tile_m = _gemm_tile_m(rows)
    return W4A16PairedKGroupGEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_aie_rows=rows // tile_m,
        tile_m=tile_m,
        tile_k=BATCH_DECODE_GEMM_TILE_K,
        tile_n=tile_n,
        k_group=2,
        context=context,
    )


def _k_group_gemm(
    context,
    *,
    rows: int,
    k: int,
    n: int,
    k_group: int = 2,
    tile_n: int = BATCH_DECODE_TRANSFORMER_TILE_N,
) -> W4A16KGroupGEMM:
    tile_m = _gemm_tile_m(rows)
    return W4A16KGroupGEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_aie_rows=rows // tile_m,
        tile_m=tile_m,
        tile_k=BATCH_DECODE_GEMM_TILE_K,
        tile_n=tile_n,
        k_group=k_group,
        context=context,
    )


def _n_shard_rows(n: int, tile_n: int) -> int:
    n_groups_per_col = n // (tile_n * BATCH_DECODE_COLUMNS)
    for rows in (4, 3, 2, 1):
        if n_groups_per_col % rows == 0:
            return rows
    raise ValueError(f"N={n} cannot be sharded across {BATCH_DECODE_COLUMNS} columns")


def _lm_head_gemm(
    context,
    *,
    rows: int,
    k: int,
    n: int,
) -> W4A16NShardGEMM:
    tile_m = _gemm_tile_m(rows)
    return W4A16NShardGEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        num_aie_rows=_n_shard_rows(n, BATCH_DECODE_LM_HEAD_TILE_N),
        tile_m=tile_m,
        tile_k=BATCH_DECODE_GEMM_TILE_K,
        tile_n=BATCH_DECODE_LM_HEAD_TILE_N,
        context=context,
    )


def build_batch_decode_fused_op(config, max_seq_len, batch_size, build_suffix):
    if config.n_layers != EXPECTED_DECODE_LAYERS:
        raise ValueError(
            f"batch decode expects {EXPECTED_DECODE_LAYERS} layers, got {config.n_layers}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size > MAX_BATCH_DECODE_ROWS:
        raise ValueError(
            f"batch_size {batch_size} exceeds padded rows {MAX_BATCH_DECODE_ROWS}"
        )
    if max_seq_len % DECODE_ATTN_CHUNK_SIZE != 0:
        raise ValueError(
            f"max_seq_len must be divisible by {DECODE_ATTN_CHUNK_SIZE}"
        )

    trace_kwargs = _trace_kwargs(build_suffix)
    build_dir_suffix = build_suffix
    if trace_kwargs:
        trace_arg = trace_kwargs["trace_ddr_id"]
        if trace_arg is None:
            trace_arg = 3
        build_dir_suffix = (
            f"{build_suffix}"
            f"_trace_op{trace_kwargs['trace_op_index']}"
            f"_arg{trace_arg}"
            f"_size{trace_kwargs['trace_size']}"
        )

    context = AIEContext(build_dir=Path("build_batch_elf") / build_dir_suffix)
    specs = _spec_by_name()
    decode_rows = select_batch_decode_rows(batch_size)

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    n_kv_groups = config.n_kv_groups
    q_heads_per_group = n_heads // n_kv_groups
    head_dim = config.head_dim
    attn_dim = n_heads * head_dim
    kv_dim = n_kv_groups * head_dim

    x_elements = decode_rows * emb_dim
    q_elements = decode_rows * attn_dim
    kv_elements = decode_rows * kv_dim
    ffn_elements = decode_rows * hidden_dim
    logits_elements = decode_rows * config.lm_head_gemm_out_features
    packet_chunk_elements = 2 * DECODE_ATTN_CHUNK_SIZE * head_dim + DECODE_ATTN_CHUNK_SIZE
    packet_elements_per_group = (
        max_seq_len // DECODE_ATTN_CHUNK_SIZE * packet_chunk_elements
    )
    packet_elements = n_kv_groups * packet_elements_per_group
    hidden_norm_columns = min(BATCH_DECODE_COLUMNS, decode_rows)
    rms_rope_columns = min(BATCH_DECODE_COLUMNS, decode_rows)

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
        num_aie_columns=hidden_norm_columns,
        num_channels=1,
        tile_size=emb_dim,
        weighted=True,
        context=context,
    )
    rms_rope_queries_op = RMSNormRoPE(
        rows=decode_rows * n_heads,
        cols=head_dim,
        angle_rows=decode_rows,
        num_aie_columns=rms_rope_columns,
        context=context,
    )
    rms_rope_keys_op = RMSNormRoPE(
        rows=decode_rows * n_kv_groups,
        cols=head_dim,
        angle_rows=decode_rows,
        num_aie_columns=rms_rope_columns,
        context=context,
    )
    gemm_attn_query_op = _k_group_gemm(
        context,
        rows=decode_rows,
        k=emb_dim,
        n=attn_dim,
        k_group=4,
    )
    gemm_attn_key_value_pair_op = _paired_k_group_gemm(
        context,
        rows=decode_rows,
        k=emb_dim,
        n=kv_dim,
    )
    gemm_attn_output_op = _k_group_gemm(
        context,
        rows=decode_rows,
        k=attn_dim,
        n=emb_dim,
    )
    gemm_ffn_up_gate_pair_op = _paired_k_group_gemm(
        context,
        rows=decode_rows,
        k=emb_dim,
        n=hidden_dim,
    )
    gemm_ffn_down_op = _k_group_gemm(
        context,
        rows=decode_rows,
        k=hidden_dim,
        n=emb_dim,
    )
    gemm_lm_head_op = _lm_head_gemm(
        context,
        rows=decode_rows,
        k=emb_dim,
        n=config.lm_head_gemm_out_features,
    )
    silu_mul_ffn_op = SiLUMul(
        size=ffn_elements,
        tile_size=hidden_dim // BATCH_DECODE_COLUMNS,
        num_aie_columns=BATCH_DECODE_COLUMNS,
        context=context,
    )
    residual_add_norm_op = ResidualAddRMSNorm(
        size=x_elements,
        tile_size=emb_dim,
        num_aie_columns=hidden_norm_columns,
        context=context,
    )
    copy_present_packet_kv_op = CopyPresentPacketKV(
        kv_dim=kv_dim,
        num_kv_groups=n_kv_groups,
        head_dim=head_dim,
        packet_elements=packet_elements,
        packet_elements_per_group=packet_elements_per_group,
        key_packet_offset=current_k_packet_offset,
        value_packet_offset=current_v_packet_offset,
        context=context,
    )
    llama_chunked_attention_op = LlamaChunkedAttention(
        max_seq_len=max_seq_len,
        num_kv_groups=n_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=DECODE_ATTN_CHUNK_SIZE,
        context=context,
    )

    runlist = [(rms_norm_op, "x", "W_norm1_0", "x_norm")]
    for layer_idx in range(config.n_layers):
        runlist.extend(
            [
                (
                    gemm_attn_query_op,
                    "x_norm",
                    _weight_arg(f"W_attn_query_{layer_idx}", specs),
                    "queries",
                ),
                (
                    gemm_attn_key_value_pair_op,
                    "x_norm",
                    f"W_attn_key_value_{layer_idx}_qparam",
                    "keys",
                    "values",
                ),
                (
                    rms_rope_queries_op,
                    "queries",
                    f"W_attn_query_norm_{layer_idx}",
                    "rope_angles",
                    "queries",
                ),
                (
                    rms_rope_keys_op,
                    "keys",
                    f"W_attn_key_norm_{layer_idx}",
                    "rope_angles",
                    "keys",
                ),
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
                        copy_present_packet_kv_op,
                        k_slice,
                        v_slice,
                        present_key_name(layer_idx, batch_idx),
                        present_value_name(layer_idx, batch_idx),
                        packet_name,
                    ),
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
                    _weight_arg(f"W_attn_output_decode_{layer_idx}", specs),
                    "attn_output",
                ),
                (
                    residual_add_norm_op,
                    "x",
                    "attn_output",
                    f"W_norm2_{layer_idx}",
                    "x",
                    "x_norm",
                ),
                (
                    gemm_ffn_up_gate_pair_op,
                    "x_norm",
                    f"W_ffn_gate_up_{layer_idx}_qparam",
                    "ffn_gate",
                    "ffn_up",
                ),
                (silu_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
                (
                    gemm_ffn_down_op,
                    "ffn_hidden",
                    _weight_arg(f"W_ffn_down_{layer_idx}", specs),
                    "ffn_output",
                ),
            ]
        )
        next_norm_weight = (
            f"W_norm1_{layer_idx + 1}"
            if layer_idx + 1 < config.n_layers
            else "W_final_norm"
        )
        next_norm_output = "x_norm" if layer_idx + 1 < config.n_layers else "hidden_out"
        runlist.append(
            (
                residual_add_norm_op,
                "x",
                "ffn_output",
                next_norm_weight,
                "x",
                next_norm_output,
            )
        )

    runlist.append(
        (
            gemm_lm_head_op,
            "hidden_out",
            _weight_arg("W_out_head", specs),
            "logits",
        )
    )

    output_args = [
        "logits",
        *[
            present_key_name(layer_idx, batch_idx)
            for layer_idx in range(config.n_layers)
            for batch_idx in range(batch_size)
        ],
        *[
            present_value_name(layer_idx, batch_idx)
            for layer_idx in range(config.n_layers)
            for batch_idx in range(batch_size)
        ],
    ]

    return (
        FusedMLIROperator(
            "quantized_batch_decode_fused_op",
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
                "logits": _bytes(logits_elements),
                **{
                    name: _bytes(packet_elements)
                    for name in batch_packet_cache_names(config, batch_size)
                },
            },
            external_args={
                "weight": batch_decode_dense_weight_names(),
                "qparam": batch_decode_qparam_names(),
                "kv_cache": batch_packet_cache_names(config, batch_size),
            },
            compile_mode="full_elf_dynamic",
            context=context,
            **trace_kwargs,
        ).compile(),
        current_cache_slot,
        decode_rows,
    )
