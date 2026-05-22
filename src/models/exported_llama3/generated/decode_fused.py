#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated decode fused operator for exported_llama3.

Regenerate with:
    uv run python -m torch2iron.export.codegen --model-package models.exported_llama3

The generator renders this file directly from torch.export.ExportedProgram.
"""

from __future__ import annotations

from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator
from torch2iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMV,
    LlamaChunkedAttention,
    RMSNorm,
    RoPE,
    SiLU,
    StridedCopy,
)

from models.exported_llama3.generated.decode_layout import (
    DECODE_LM_HEAD_WEIGHT_NAMES,
    DECODE_OUTPUT_ARGS,
    DECODE_PACKET_CACHE_NAMES,
    DECODE_TRANSFORMER_WEIGHT_NAMES,
    EXPECTED_DECODE_LAYERS,
    EXPORTED_DECODE_CHUNK_SIZE,
)
from models.exported_llama3.runtime_config import DECODE_ATTN_CHUNK_SIZE


def build_decode_fused_op(config, prompt_len, build_suffix):
    if config.n_layers != EXPECTED_DECODE_LAYERS:
        raise ValueError(
            f"generated decode graph expects {EXPECTED_DECODE_LAYERS} layers, "
            f"got {config.n_layers}"
        )
    if DECODE_ATTN_CHUNK_SIZE != EXPORTED_DECODE_CHUNK_SIZE:
        raise ValueError(
            f"generated decode graph expects chunk size {EXPORTED_DECODE_CHUNK_SIZE}, "
            f"got {DECODE_ATTN_CHUNK_SIZE}"
        )

    elf_ctx = AIEContext(build_dir=Path("build_elf") / build_suffix)

    gemv_attn_query_op = GEMV(
        M=config.n_heads * config.head_dim,
        K=config.emb_dim,
        num_aie_columns=8,
        tile_size_input=4,
        tile_size_output=config.head_dim // 2,
        context=elf_ctx,
    )

    gemv_attn_key_value_op = GEMV(
        M=config.n_kv_groups * config.head_dim,
        K=config.emb_dim,
        num_aie_columns=8,
        tile_size_input=4,
        tile_size_output=config.head_dim // 2,
        context=elf_ctx,
    )

    rope_queries_op = RoPE(
        rows=config.n_heads,
        cols=config.head_dim,
        angle_rows=1,
        context=elf_ctx,
    )
    rope_keys_op = RoPE(
        rows=config.n_kv_groups,
        cols=config.head_dim,
        angle_rows=1,
        context=elf_ctx,
    )

    current_cache_slot = prompt_len - 1
    packet_chunk_elements = (
        2 * DECODE_ATTN_CHUNK_SIZE * config.head_dim + DECODE_ATTN_CHUNK_SIZE
    )
    packet_elements_per_group = (
        prompt_len // DECODE_ATTN_CHUNK_SIZE * packet_chunk_elements
    )
    packet_elements = config.n_kv_groups * packet_elements_per_group
    current_chunk = current_cache_slot // DECODE_ATTN_CHUNK_SIZE
    current_row = current_cache_slot % DECODE_ATTN_CHUNK_SIZE
    current_k_packet_offset = (
        current_chunk * packet_chunk_elements + current_row * config.head_dim
    )
    current_v_packet_offset = (
        current_chunk * packet_chunk_elements
        + DECODE_ATTN_CHUNK_SIZE * config.head_dim
        + current_row * config.head_dim
    )

    strided_copy_packet_key_op = StridedCopy(
        input_sizes=(config.n_kv_groups, config.head_dim),
        input_strides=(config.head_dim, 1),
        input_offset=0,
        output_sizes=(config.n_kv_groups, config.head_dim),
        output_strides=(packet_elements_per_group, 1),
        output_offset=current_k_packet_offset,
        input_buffer_size=config.n_kv_groups * config.head_dim,
        output_buffer_size=packet_elements,
        num_aie_channels=1,
        context=elf_ctx,
    )

    strided_copy_packet_value_op = StridedCopy(
        input_sizes=(config.n_kv_groups, config.head_dim),
        input_strides=(config.head_dim, 1),
        input_offset=0,
        output_sizes=(config.n_kv_groups, config.head_dim),
        output_strides=(packet_elements_per_group, 1),
        output_offset=current_v_packet_offset,
        input_buffer_size=config.n_kv_groups * config.head_dim,
        output_buffer_size=packet_elements,
        num_aie_channels=1,
        context=elf_ctx,
    )

    copy_present_kv_op = StridedCopy(
        input_sizes=(config.n_kv_groups, config.head_dim),
        input_strides=(config.head_dim, 1),
        input_offset=0,
        output_sizes=(config.n_kv_groups, config.head_dim),
        output_strides=(config.head_dim, 1),
        output_offset=0,
        input_buffer_size=config.n_kv_groups * config.head_dim,
        output_buffer_size=config.n_kv_groups * config.head_dim,
        num_aie_channels=1,
        context=elf_ctx,
    )

    llama_chunked_attention_op = LlamaChunkedAttention(
        max_seq_len=prompt_len,
        num_kv_groups=config.n_kv_groups,
        q_heads_per_group=config.n_heads // config.n_kv_groups,
        head_dim=config.head_dim,
        chunk_size=DECODE_ATTN_CHUNK_SIZE,
        context=elf_ctx,
    )

    gemv_attn_output_op = GEMV(
        M=config.emb_dim,
        K=config.n_heads * config.head_dim,
        num_aie_columns=8,
        tile_size_input=4,
        tile_size_output=config.emb_dim // 8,
        context=elf_ctx,
    )

    rms_norm_op = RMSNorm(
        size=config.emb_dim,
        num_aie_columns=1,
        num_channels=1,
        tile_size=config.emb_dim,
        weighted=True,
        context=elf_ctx,
    )

    gemv_ffn_up_gate_op = GEMV(
        M=config.hidden_dim,
        K=config.emb_dim,
        num_aie_columns=8,
        tile_size_input=4,
        tile_size_output=config.hidden_dim // 8,
        context=elf_ctx,
    )

    gemv_ffn_down_op = GEMV(
        M=config.emb_dim,
        K=config.hidden_dim,
        num_aie_columns=8,
        tile_size_input=1,
        tile_size_output=config.emb_dim // 8,
        context=elf_ctx,
    )

    silu_ffn_op = SiLU(
        size=config.hidden_dim,
        tile_size=config.hidden_dim // 8,
        num_aie_columns=8,
        context=elf_ctx,
    )

    eltwise_mul_ffn_op = ElementwiseMul(
        size=config.hidden_dim,
        tile_size=config.hidden_dim // 8,
        num_aie_columns=8,
        context=elf_ctx,
    )

    residual_add_op = ElementwiseAdd(
        size=config.emb_dim,
        tile_size=config.emb_dim // 8,
        context=elf_ctx,
    )

    gemv_out_head_op = GEMV(
        M=config.vocab_size,
        K=config.emb_dim,
        num_aie_columns=8,
        tile_size_input=4,
        tile_size_output=32,
        context=elf_ctx,
    )

    packet_cache_buffer_size = packet_elements * 2
    runlist = [
        (rms_norm_op, "x", "W_norm1_0", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_0", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_0", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_0", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_0"),
        (copy_present_kv_op, "values", "present_values_0"),
        (strided_copy_packet_key_op, "keys", "packet_cache_0"),
        (strided_copy_packet_value_op, "values", "packet_cache_0"),
        (llama_chunked_attention_op, "queries", "packet_cache_0", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_0", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_0", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_0", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_0", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_0", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_1", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_1", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_1", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_1", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_1"),
        (copy_present_kv_op, "values", "present_values_1"),
        (strided_copy_packet_key_op, "keys", "packet_cache_1"),
        (strided_copy_packet_value_op, "values", "packet_cache_1"),
        (llama_chunked_attention_op, "queries", "packet_cache_1", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_1", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_1", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_1", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_1", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_1", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_2", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_2", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_2", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_2", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_2"),
        (copy_present_kv_op, "values", "present_values_2"),
        (strided_copy_packet_key_op, "keys", "packet_cache_2"),
        (strided_copy_packet_value_op, "values", "packet_cache_2"),
        (llama_chunked_attention_op, "queries", "packet_cache_2", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_2", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_2", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_2", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_2", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_2", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_3", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_3", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_3", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_3", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_3"),
        (copy_present_kv_op, "values", "present_values_3"),
        (strided_copy_packet_key_op, "keys", "packet_cache_3"),
        (strided_copy_packet_value_op, "values", "packet_cache_3"),
        (llama_chunked_attention_op, "queries", "packet_cache_3", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_3", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_3", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_3", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_3", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_3", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_4", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_4", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_4", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_4", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_4"),
        (copy_present_kv_op, "values", "present_values_4"),
        (strided_copy_packet_key_op, "keys", "packet_cache_4"),
        (strided_copy_packet_value_op, "values", "packet_cache_4"),
        (llama_chunked_attention_op, "queries", "packet_cache_4", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_4", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_4", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_4", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_4", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_4", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_5", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_5", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_5", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_5", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_5"),
        (copy_present_kv_op, "values", "present_values_5"),
        (strided_copy_packet_key_op, "keys", "packet_cache_5"),
        (strided_copy_packet_value_op, "values", "packet_cache_5"),
        (llama_chunked_attention_op, "queries", "packet_cache_5", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_5", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_5", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_5", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_5", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_5", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_6", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_6", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_6", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_6", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_6"),
        (copy_present_kv_op, "values", "present_values_6"),
        (strided_copy_packet_key_op, "keys", "packet_cache_6"),
        (strided_copy_packet_value_op, "values", "packet_cache_6"),
        (llama_chunked_attention_op, "queries", "packet_cache_6", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_6", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_6", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_6", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_6", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_6", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_7", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_7", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_7", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_7", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_7"),
        (copy_present_kv_op, "values", "present_values_7"),
        (strided_copy_packet_key_op, "keys", "packet_cache_7"),
        (strided_copy_packet_value_op, "values", "packet_cache_7"),
        (llama_chunked_attention_op, "queries", "packet_cache_7", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_7", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_7", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_7", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_7", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_7", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_8", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_8", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_8", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_8", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_8"),
        (copy_present_kv_op, "values", "present_values_8"),
        (strided_copy_packet_key_op, "keys", "packet_cache_8"),
        (strided_copy_packet_value_op, "values", "packet_cache_8"),
        (llama_chunked_attention_op, "queries", "packet_cache_8", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_8", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_8", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_8", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_8", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_8", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_9", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_9", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_9", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_9", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_9"),
        (copy_present_kv_op, "values", "present_values_9"),
        (strided_copy_packet_key_op, "keys", "packet_cache_9"),
        (strided_copy_packet_value_op, "values", "packet_cache_9"),
        (llama_chunked_attention_op, "queries", "packet_cache_9", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_9", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_9", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_9", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_9", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_9", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_10", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_10", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_10", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_10", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_10"),
        (copy_present_kv_op, "values", "present_values_10"),
        (strided_copy_packet_key_op, "keys", "packet_cache_10"),
        (strided_copy_packet_value_op, "values", "packet_cache_10"),
        (llama_chunked_attention_op, "queries", "packet_cache_10", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_10", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_10", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_10", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_10", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_10", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_11", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_11", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_11", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_11", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_11"),
        (copy_present_kv_op, "values", "present_values_11"),
        (strided_copy_packet_key_op, "keys", "packet_cache_11"),
        (strided_copy_packet_value_op, "values", "packet_cache_11"),
        (llama_chunked_attention_op, "queries", "packet_cache_11", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_11", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_11", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_11", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_11", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_11", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_12", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_12", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_12", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_12", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_12"),
        (copy_present_kv_op, "values", "present_values_12"),
        (strided_copy_packet_key_op, "keys", "packet_cache_12"),
        (strided_copy_packet_value_op, "values", "packet_cache_12"),
        (llama_chunked_attention_op, "queries", "packet_cache_12", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_12", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_12", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_12", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_12", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_12", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_13", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_13", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_13", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_13", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_13"),
        (copy_present_kv_op, "values", "present_values_13"),
        (strided_copy_packet_key_op, "keys", "packet_cache_13"),
        (strided_copy_packet_value_op, "values", "packet_cache_13"),
        (llama_chunked_attention_op, "queries", "packet_cache_13", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_13", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_13", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_13", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_13", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_13", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_14", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_14", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_14", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_14", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_14"),
        (copy_present_kv_op, "values", "present_values_14"),
        (strided_copy_packet_key_op, "keys", "packet_cache_14"),
        (strided_copy_packet_value_op, "values", "packet_cache_14"),
        (llama_chunked_attention_op, "queries", "packet_cache_14", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_14", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_14", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_14", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_14", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_14", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_norm1_15", "x_norm"),
        (gemv_attn_query_op, "W_attn_query_15", "x_norm", "queries"),
        (gemv_attn_key_value_op, "W_attn_key_15", "x_norm", "keys"),
        (gemv_attn_key_value_op, "W_attn_value_15", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", "present_keys_15"),
        (copy_present_kv_op, "values", "present_values_15"),
        (strided_copy_packet_key_op, "keys", "packet_cache_15"),
        (strided_copy_packet_value_op, "values", "packet_cache_15"),
        (llama_chunked_attention_op, "queries", "packet_cache_15", "attn_context"),
        (gemv_attn_output_op, "W_attn_output_decode_15", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", "W_norm2_15", "x_norm"),
        (gemv_ffn_up_gate_op, "W_ffn_gate_15", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, "W_ffn_up_15", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, "W_ffn_down_15", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
        (rms_norm_op, "x", "W_final_norm", "hidden_out"),
        (gemv_out_head_op, "W_out_head", "hidden_out", "logits"),
    ]

    fused_op = FusedMLIROperator(
        "fused_op",
        runlist,
        input_args=["x", "rope_angles"],
        output_args=list(DECODE_OUTPUT_ARGS),
        buffer_sizes={
            name: packet_cache_buffer_size for name in DECODE_PACKET_CACHE_NAMES
        },
        external_args={
            "weight": list(DECODE_TRANSFORMER_WEIGHT_NAMES),
            "lm_head": list(DECODE_LM_HEAD_WEIGHT_NAMES),
            "kv_cache": list(DECODE_PACKET_CACHE_NAMES),
        },
        context=elf_ctx,
    ).compile()
    return fused_op, current_cache_slot
