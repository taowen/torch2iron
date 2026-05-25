#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for fast Qwen3 decode packet-cache attention."""

from __future__ import annotations

import math

import torch


def decode_packet_chunk_elements(chunk_size: int, head_dim: int) -> int:
    return 2 * chunk_size * head_dim + chunk_size


def decode_packet_elements_per_group(
    packet_seq_len: int,
    chunk_size: int,
    head_dim: int,
) -> int:
    if packet_seq_len % chunk_size != 0:
        raise ValueError("packet_seq_len must be divisible by chunk_size")
    return (
        packet_seq_len // chunk_size
        * decode_packet_chunk_elements(chunk_size, head_dim)
    )


def decode_packet_elements(
    num_kv_groups: int,
    packet_seq_len: int,
    chunk_size: int,
    head_dim: int,
) -> int:
    return num_kv_groups * decode_packet_elements_per_group(
        packet_seq_len,
        chunk_size,
        head_dim,
    )


def decode_packet_slot_offsets(
    group_idx: int,
    slot: int,
    packet_seq_len: int,
    chunk_size: int,
    head_dim: int,
) -> tuple[int, int, int]:
    if slot < 0 or slot >= packet_seq_len:
        raise ValueError("slot must be inside packet_seq_len")
    chunk_idx = slot // chunk_size
    row = slot % chunk_size
    chunk_elements = decode_packet_chunk_elements(chunk_size, head_dim)
    group_base = group_idx * decode_packet_elements_per_group(
        packet_seq_len,
        chunk_size,
        head_dim,
    )
    chunk_base = group_base + chunk_idx * chunk_elements
    key_offset = chunk_base + row * head_dim
    value_offset = chunk_base + chunk_size * head_dim + row * head_dim
    mask_offset = chunk_base + 2 * chunk_size * head_dim + row
    return key_offset, value_offset, mask_offset


def write_decode_packet_slot(
    packet: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    slot: int,
    packet_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    if packet.dim() != 1:
        raise ValueError("packet must be a flat tensor")
    if current_key.shape != current_value.shape or current_key.dim() != 2:
        raise ValueError("current_key/current_value must be [groups, head_dim]")

    updated = packet.clone()
    num_kv_groups = int(current_key.shape[0])
    head_dim = int(current_key.shape[1])
    expected_elements = decode_packet_elements(
        num_kv_groups,
        packet_seq_len,
        chunk_size,
        head_dim,
    )
    if packet.numel() != expected_elements:
        raise ValueError(
            f"packet has {packet.numel()} elements, expected {expected_elements}"
        )

    for group_idx in range(num_kv_groups):
        key_offset, value_offset, mask_offset = decode_packet_slot_offsets(
            group_idx,
            slot,
            packet_seq_len,
            chunk_size,
            head_dim,
        )
        updated[key_offset : key_offset + head_dim] = current_key[group_idx]
        updated[value_offset : value_offset + head_dim] = current_value[group_idx]
        updated[mask_offset] = torch.tensor(1.0, dtype=updated.dtype)
    return updated


def chunked_attention_reference(
    queries: torch.Tensor,
    packet: torch.Tensor,
    attend_seq_len: int,
    packet_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    if queries.dim() != 3:
        raise ValueError("queries must be [groups, q_heads_per_group, head_dim]")
    if packet.dim() != 1:
        raise ValueError("packet must be flat")
    if attend_seq_len <= 0 or attend_seq_len > packet_seq_len:
        raise ValueError("attend_seq_len must be in (0, packet_seq_len]")
    if attend_seq_len % chunk_size != 0:
        raise ValueError("attend_seq_len must be divisible by chunk_size")

    num_kv_groups = int(queries.shape[0])
    q_heads_per_group = int(queries.shape[1])
    head_dim = int(queries.shape[2])
    expected_elements = decode_packet_elements(
        num_kv_groups,
        packet_seq_len,
        chunk_size,
        head_dim,
    )
    if packet.numel() != expected_elements:
        raise ValueError(
            f"packet has {packet.numel()} elements, expected {expected_elements}"
        )

    output = torch.empty_like(queries)
    scale = 1.0 / math.sqrt(head_dim)
    for group_idx in range(num_kv_groups):
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for slot in range(attend_seq_len):
            key_offset, value_offset, mask_offset = decode_packet_slot_offsets(
                group_idx,
                slot,
                packet_seq_len,
                chunk_size,
                head_dim,
            )
            if float(packet[mask_offset].item()) <= 0.5:
                continue
            keys.append(packet[key_offset : key_offset + head_dim].to(torch.float32))
            values.append(packet[value_offset : value_offset + head_dim].to(torch.float32))

        if not keys:
            output[group_idx].zero_()
            continue

        key_matrix = torch.stack(keys, dim=0)
        value_matrix = torch.stack(values, dim=0)
        query_matrix = queries[group_idx].to(torch.float32)
        scores = query_matrix.matmul(key_matrix.t()) * scale
        weights = torch.softmax(scores, dim=-1)
        context = weights.matmul(value_matrix)
        if context.shape != (q_heads_per_group, head_dim):
            raise AssertionError("unexpected attention context shape")
        output[group_idx] = context.to(torch.bfloat16)
    return output


def chunked_attention_update_reference(
    queries: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    packet: torch.Tensor,
    current_slot: int,
    attend_seq_len: int,
    packet_seq_len: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_packet = write_decode_packet_slot(
        packet,
        current_key,
        current_value,
        current_slot,
        packet_seq_len,
        chunk_size,
    )
    context = chunked_attention_reference(
        queries,
        updated_packet,
        attend_seq_len,
        packet_seq_len,
        chunk_size,
    )
    return context, updated_packet
