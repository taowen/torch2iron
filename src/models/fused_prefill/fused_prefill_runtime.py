#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime wrapper for chunked fused prefill."""

from __future__ import annotations

import torch

from models.fused_prefill.decode_packet_cache import (
    decode_packet_slot_offsets,
    sync_decode_packet_range,
)
from models.fused_prefill.generated.decode_layout import DECODE_PACKET_CACHE_NAMES
from models.fused_prefill.runtime_config import (
    PREFILL_CHUNK_COMPUTE_ROWS,
    PREFILL_CHUNK_SIZE,
    PREFILL_LM_HEAD_ROWS,
)


def _zero_prefill_packet_caches(config, fused):
    for layer_idx in range(config.n_layers):
        packet_cache = fused.get_buffer(DECODE_PACKET_CACHE_NAMES[layer_idx])
        packet_cache.torch_view().zero_()
        packet_cache.to("npu")


def _append_prefill_chunk_to_packet_cache(
    config,
    max_seq_len,
    fused,
    layer_idx,
    chunk_start,
    valid_len,
):
    present_key = (
        fused.get_buffer(f"present_keys_{layer_idx}")
        .torch_view()
        .view(config.n_kv_groups, PREFILL_CHUNK_SIZE, config.head_dim)
    )
    present_value = (
        fused.get_buffer(f"present_values_{layer_idx}")
        .torch_view()
        .view(config.n_kv_groups, PREFILL_CHUNK_SIZE, config.head_dim)
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


def _run_lm_head_for_last_chunk(runner, hidden_out, valid_len):
    config = runner.config
    lm_head = runner.aie_ops.prefill.lm_head

    x = lm_head.get_buffer("x").torch_view().view(PREFILL_LM_HEAD_ROWS, config.emb_dim)
    x.zero_()
    x[:valid_len, :] = hidden_out[:valid_len, :]

    lm_head()
    partition_width = config.padded_vocab_size // config.vocab_partitions
    logits_padded = torch.cat(
        [
            lm_head.get_buffer(f"logits_part_{part_idx}").torch_view().view(
                PREFILL_LM_HEAD_ROWS,
                partition_width,
            )
            for part_idx in range(config.vocab_partitions)
        ],
        dim=1,
    )
    return logits_padded.unsqueeze(0)[:, :valid_len, : config.vocab_size]


def fused_prefill_forward_pass(runner, state):
    config = runner.config
    max_seq_len = runner.max_seq_len
    fused = runner.aie_ops.prefill.fused

    _, seq_len = state.token_ids.shape
    if seq_len > max_seq_len:
        raise ValueError(f"prefill seq_len {seq_len} exceeds static {max_seq_len}")

    if state.num_preceding_tokens:
        raise NotImplementedError("chunked fused prefill only supports first prefill")

    _zero_prefill_packet_caches(config, fused)

    tok_emb_weight = config.weights["model.embed_tokens.weight"]
    last_logits = None

    for chunk_start in range(0, seq_len, PREFILL_CHUNK_SIZE):
        valid_len = min(PREFILL_CHUNK_SIZE, seq_len - chunk_start)
        chunk_end = chunk_start + valid_len

        rope_angles = fused.get_buffer("rope_angles").torch_view().view(
            PREFILL_CHUNK_COMPUTE_ROWS, config.head_dim
        )
        rope_angles.zero_()
        rope_angles[:valid_len, :] = config.angles[chunk_start:chunk_end]

        chunk_token_ids = state.token_ids[:, chunk_start:chunk_end]
        x = torch.nn.functional.embedding(chunk_token_ids, tok_emb_weight)
        x_input = fused.get_buffer("x").torch_view().view(
            PREFILL_CHUNK_COMPUTE_ROWS, config.emb_dim
        )
        x_input.zero_()
        x_input[:valid_len, :] = x[0, :, :]

        fused()

        for layer_idx in range(config.n_layers):
            _append_prefill_chunk_to_packet_cache(
                config,
                max_seq_len,
                fused,
                layer_idx,
                chunk_start,
                valid_len,
            )

        if chunk_end == seq_len:
            hidden_out = fused.get_buffer("hidden_out").torch_view().view(
                PREFILL_CHUNK_COMPUTE_ROWS, config.emb_dim
            )
            last_logits = _run_lm_head_for_last_chunk(runner, hidden_out, valid_len)

    if last_logits is None:
        raise RuntimeError("prefill produced no logits")

    return last_logits, state
