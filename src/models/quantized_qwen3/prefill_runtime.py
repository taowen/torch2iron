#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime wrapper for chunked fused prefill."""

from __future__ import annotations

import torch

from models.quantized_qwen3.decode_packet_cache import (
    decode_packet_slot_offsets,
    sync_decode_packet_range,
)
from models.quantized_qwen3.generated.decode_layout import DECODE_PACKET_CACHE_NAMES
from models.quantized_qwen3.runtime_config import DECODE_ATTN_CHUNK_SIZE


def _clear_prefill_packet_masks(config, max_seq_len, fused):
    for layer_idx in range(config.n_layers):
        packet_cache = fused.get_buffer(DECODE_PACKET_CACHE_NAMES[layer_idx])
        packet = packet_cache.torch_view()
        for group_idx in range(config.n_kv_groups):
            for slot in range(0, max_seq_len, DECODE_ATTN_CHUNK_SIZE):
                _k_offset, _v_offset, mask_offset = decode_packet_slot_offsets(
                    config,
                    max_seq_len,
                    group_idx,
                    slot,
                    DECODE_ATTN_CHUNK_SIZE,
                )
                rows = min(DECODE_ATTN_CHUNK_SIZE, max_seq_len - slot)
                packet[mask_offset : mask_offset + rows] = 0.0
                sync_decode_packet_range(packet_cache, mask_offset, rows)


def _append_prefill_chunk_to_packet_cache(
    config,
    max_seq_len,
    fused,
    layer_idx,
    chunk_start,
    valid_len,
    chunk_size,
):
    present_key = (
        fused.get_buffer(f"present_keys_{layer_idx}")
        .torch_view()
        .view(config.n_kv_groups, chunk_size, config.head_dim)
    )
    present_value = (
        fused.get_buffer(f"present_values_{layer_idx}")
        .torch_view()
        .view(config.n_kv_groups, chunk_size, config.head_dim)
    )
    packet_cache = fused.get_buffer(DECODE_PACKET_CACHE_NAMES[layer_idx])
    packet = packet_cache.torch_view()

    for group_idx in range(config.n_kv_groups):
        k_offset, v_offset, mask_offset = decode_packet_slot_offsets(
            config, max_seq_len, group_idx, chunk_start
        )
        packet[k_offset : k_offset + valid_len * config.head_dim] = present_key[
            group_idx, :valid_len, :
        ].reshape(-1)
        packet[v_offset : v_offset + valid_len * config.head_dim] = present_value[
            group_idx, :valid_len, :
        ].reshape(-1)
        packet[mask_offset : mask_offset + valid_len] = 1.0

        sync_decode_packet_range(
            packet_cache,
            k_offset,
            mask_offset + valid_len - k_offset,
        )


def _select_prefill_elf(prefill_ops, *, final_chunk: bool):
    fused = prefill_ops.fused
    elf_kind = "final" if final_chunk else "body"
    has_run = getattr(prefill_ops, "_prefill_dispatch_has_run", False)
    if prefill_ops.loaded_elf_kind == elf_kind and not has_run:
        return
    elf_data = prefill_ops.final_elf_data if final_chunk else prefill_ops.body_elf_data
    fused.reload_elf(elf_data)
    prefill_ops.loaded_elf_kind = elf_kind
    fused.weight_buffer.to("npu")
    fused.qparam_buffer.to("npu")
    fused.kv_cache_buffer.to("npu")
    fused.scratch_buffer.to("npu")
    fused.output_buffer.to("npu")


def prefill_forward_pass(runner, state):
    config = runner.config
    max_seq_len = runner.prefill_max_seq_len
    prefill_ops = runner.aie_ops.prefill
    fused = prefill_ops.fused
    chunk_size = prefill_ops.chunk_size
    compute_rows = prefill_ops.compute_rows

    _, seq_len = state.token_ids.shape
    if seq_len > max_seq_len:
        raise ValueError(f"prefill seq_len {seq_len} exceeds static {max_seq_len}")

    if state.num_preceding_tokens:
        raise NotImplementedError("chunked fused prefill only supports first prefill")

    _clear_prefill_packet_masks(config, max_seq_len, fused)

    tok_emb_weight = config.weights["model.embed_tokens.weight"]
    last_logits = None

    for chunk_start in range(0, seq_len, chunk_size):
        valid_len = min(chunk_size, seq_len - chunk_start)
        chunk_end = chunk_start + valid_len
        final_chunk = chunk_end == seq_len

        fused.mark_buffer_dirty("input")
        rope_angles = fused.get_buffer("rope_angles").torch_view().view(
            compute_rows, config.head_dim
        )
        rope_angles.zero_()
        rope_angles[:valid_len, :] = config.angles[chunk_start:chunk_end]

        chunk_token_ids = state.token_ids[:, chunk_start:chunk_end]
        x = torch.nn.functional.embedding(chunk_token_ids, tok_emb_weight)
        x_input = fused.get_buffer("x").torch_view().view(
            compute_rows, config.emb_dim
        )
        x_input.zero_()
        x_input[:valid_len, :] = x[0, :, :]

        _select_prefill_elf(prefill_ops, final_chunk=final_chunk)
        fused()
        prefill_ops._prefill_dispatch_has_run = True

        for layer_idx in range(config.n_layers):
            _append_prefill_chunk_to_packet_cache(
                config,
                max_seq_len,
                fused,
                layer_idx,
                chunk_start,
                valid_len,
                chunk_size,
            )

        if final_chunk:
            logits = fused.get_buffer("logits").torch_view().view(
                compute_rows,
                config.lm_head_gemm_out_features,
            )
            last_logits = logits[valid_len - 1 : valid_len, : config.vocab_size].view(
                1,
                1,
                config.vocab_size,
            )

    if last_logits is None:
        raise RuntimeError("prefill produced no logits")

    return last_logits, state
