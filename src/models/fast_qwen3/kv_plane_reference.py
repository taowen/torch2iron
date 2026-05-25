#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the FastFlowLM-style Qwen KV plane layout."""

from __future__ import annotations

import math

import torch


def kv_plane_token_stride_elements(head_dim: int) -> int:
    return 4 * head_dim


def kv_plane_elements(packet_seq_len: int, head_dim: int) -> int:
    return packet_seq_len * kv_plane_token_stride_elements(head_dim)


def kv_plane_total_elements(packet_seq_len: int, head_dim: int) -> int:
    return 4 * kv_plane_elements(packet_seq_len, head_dim)


def kv_plane_group_offsets(
    group_idx: int,
    slot: int,
    packet_seq_len: int,
    head_dim: int,
) -> tuple[int, int]:
    if group_idx < 0 or group_idx >= 8:
        raise ValueError("group_idx must be in [0, 8)")
    if slot < 0 or slot >= packet_seq_len:
        raise ValueError("slot must be inside packet_seq_len")

    plane_group_count = 4
    plane_elements = kv_plane_elements(packet_seq_len, head_dim)
    token_base = slot * kv_plane_token_stride_elements(head_dim)
    if group_idx < plane_group_count:
        key_plane = 0
        value_plane = 1
        group_in_plane = group_idx
    else:
        key_plane = 2
        value_plane = 3
        group_in_plane = group_idx - plane_group_count

    row_offset = token_base + group_in_plane * head_dim
    return (
        key_plane * plane_elements + row_offset,
        value_plane * plane_elements + row_offset,
    )


def write_q_current_to_kv_plane(
    q_current: torch.Tensor,
    current_slot: int,
    packet_seq_len: int,
    q_heads_per_group: int,
) -> torch.Tensor:
    head_dim = int(q_current.shape[1]) // (q_heads_per_group + 2)
    kv_plane = torch.zeros(
        kv_plane_total_elements(packet_seq_len, head_dim),
        dtype=q_current.dtype,
    )
    update_q_current_in_kv_plane(
        q_current,
        kv_plane,
        current_slot,
        packet_seq_len,
        q_heads_per_group,
    )
    return kv_plane


def update_q_current_in_kv_plane(
    q_current: torch.Tensor,
    kv_plane: torch.Tensor,
    current_slot: int,
    packet_seq_len: int,
    q_heads_per_group: int,
) -> None:
    if q_current.dim() != 2:
        raise ValueError("q_current must be [8, q_current_elements_per_group]")
    if int(q_current.shape[0]) != 8:
        raise ValueError("q_current must contain exactly 8 KV groups")
    if kv_plane.dim() != 1:
        raise ValueError("kv_plane must be flat")

    head_dim = int(q_current.shape[1]) // (q_heads_per_group + 2)
    q_elements_per_group = q_heads_per_group * head_dim
    expected_group_elements = q_elements_per_group + 2 * head_dim
    if int(q_current.shape[1]) != expected_group_elements:
        raise ValueError("q_current shape does not match q_heads_per_group")
    expected_plane_elements = kv_plane_total_elements(packet_seq_len, head_dim)
    if int(kv_plane.numel()) != expected_plane_elements:
        raise ValueError(
            f"kv_plane has {kv_plane.numel()} elements, expected {expected_plane_elements}"
        )

    for group_idx in range(8):
        key_dst, value_dst = kv_plane_group_offsets(
            group_idx,
            current_slot,
            packet_seq_len,
            head_dim,
        )
        key_src = q_elements_per_group
        value_src = key_src + head_dim
        kv_plane[key_dst : key_dst + head_dim] = q_current[
            group_idx,
            key_src : key_src + head_dim,
        ]
        kv_plane[value_dst : value_dst + head_dim] = q_current[
            group_idx,
            value_src : value_src + head_dim,
        ]


def plane_attention_current_reference(
    q_current: torch.Tensor,
    kv_plane: torch.Tensor,
    current_slot: int,
    attend_seq_len: int,
    packet_seq_len: int,
    q_heads_per_group: int,
) -> torch.Tensor:
    if q_current.dim() != 2:
        raise ValueError("q_current must be [8, q_current_elements_per_group]")
    if int(q_current.shape[0]) != 8:
        raise ValueError("q_current must contain exactly 8 KV groups")
    if kv_plane.dim() != 1:
        raise ValueError("kv_plane must be flat")
    if attend_seq_len <= 0 or attend_seq_len > packet_seq_len:
        raise ValueError("attend_seq_len must be in (0, packet_seq_len]")
    if current_slot < 0 or current_slot >= attend_seq_len:
        raise ValueError("current_slot must be inside attend_seq_len")

    head_dim = int(q_current.shape[1]) // (q_heads_per_group + 2)
    expected_elements = kv_plane_total_elements(packet_seq_len, head_dim)
    if int(kv_plane.numel()) != expected_elements:
        raise ValueError(
            f"kv_plane has {kv_plane.numel()} elements, expected {expected_elements}"
        )

    q_elements_per_group = q_heads_per_group * head_dim
    output = torch.empty(8, q_heads_per_group, head_dim, dtype=q_current.dtype)
    scale = 1.0 / math.sqrt(head_dim)
    for group_idx in range(8):
        current_key = q_current[
            group_idx,
            q_elements_per_group : q_elements_per_group + head_dim,
        ].to(torch.float32)
        current_value = q_current[
            group_idx,
            q_elements_per_group + head_dim : q_elements_per_group + 2 * head_dim,
        ].to(torch.float32)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for slot in range(attend_seq_len):
            if slot == current_slot:
                keys.append(current_key)
                values.append(current_value)
                continue
            key_offset, value_offset = kv_plane_group_offsets(
                group_idx,
                slot,
                packet_seq_len,
                head_dim,
            )
            keys.append(kv_plane[key_offset : key_offset + head_dim].to(torch.float32))
            values.append(
                kv_plane[value_offset : value_offset + head_dim].to(torch.float32)
            )

        key_matrix = torch.stack(keys, dim=0)
        value_matrix = torch.stack(values, dim=0)
        query_matrix = q_current[group_idx, :q_elements_per_group].view(
            q_heads_per_group,
            head_dim,
        ).to(torch.float32)
        scores = query_matrix.matmul(key_matrix.t()) * scale
        weights = torch.softmax(scores, dim=-1)
        output[group_idx] = weights.matmul(value_matrix).to(torch.bfloat16)
    return output
