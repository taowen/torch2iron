#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference fused Q/K/V projection for the fast Qwen3 layer path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.fast_qwen3.fast_packed_format import FastQwen3Store


@dataclass(frozen=True)
class FusedQKVOutput:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


def layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def qkv_prefixes(layer_idx: int) -> tuple[str, str, str]:
    prefix = layer_prefix(layer_idx)
    return (
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.v_proj",
    )


def fused_qkv_reference(
    store: FastQwen3Store,
    layer_idx: int,
    hidden: torch.Tensor,
) -> FusedQKVOutput:
    q_prefix, k_prefix, v_prefix = qkv_prefixes(layer_idx)
    return FusedQKVOutput(
        query=store.linear_reference(q_prefix, hidden),
        key=store.linear_reference(k_prefix, hidden),
        value=store.linear_reference(v_prefix, hidden),
    )


def qkv_output_shapes(output: FusedQKVOutput) -> dict[str, tuple[int, ...]]:
    return {
        "query": tuple(int(dim) for dim in output.query.shape),
        "key": tuple(int(dim) for dim in output.key.shape),
        "value": tuple(int(dim) for dim in output.value.shape),
    }


def assemble_q_current_from_qkv_patches(
    qkv_patches: torch.Tensor,
    num_kv_groups: int,
    q_heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    if qkv_patches.dim() != 3 or qkv_patches.shape[1] != 3:
        raise ValueError("qkv_patches must be [patches, 3, patch_rows]")
    patch_rows = int(qkv_patches.shape[2])
    if head_dim % patch_rows != 0:
        raise ValueError("head_dim must be divisible by qkv patch rows")
    head_chunks = head_dim // patch_rows
    q_elements_per_group = q_heads_per_group * head_dim
    q_current = torch.empty(
        (num_kv_groups, q_elements_per_group + 2 * head_dim),
        dtype=qkv_patches.dtype,
    )
    required_q_patches = num_kv_groups * q_heads_per_group * head_chunks
    required_kv_patches = num_kv_groups * head_chunks
    if qkv_patches.shape[0] < required_q_patches:
        raise ValueError("qkv_patches do not cover requested Q groups")
    if qkv_patches.shape[0] < required_kv_patches:
        raise ValueError("qkv_patches do not cover requested K/V groups")

    for group_idx in range(num_kv_groups):
        q_base = group_idx * q_heads_per_group * head_chunks
        kv_base = group_idx * head_chunks
        for q_head in range(q_heads_per_group):
            for head_chunk in range(head_chunks):
                src_patch = q_base + q_head * head_chunks + head_chunk
                dst_start = q_head * head_dim + head_chunk * patch_rows
                q_current[group_idx, dst_start : dst_start + patch_rows] = qkv_patches[
                    src_patch, 0
                ]
        key_base = q_elements_per_group
        value_base = q_elements_per_group + head_dim
        for head_chunk in range(head_chunks):
            src_patch = kv_base + head_chunk
            dst_start = head_chunk * patch_rows
            q_current[group_idx, key_base + dst_start : key_base + dst_start + patch_rows] = (
                qkv_patches[src_patch, 1]
            )
            q_current[
                group_idx,
                value_base + dst_start : value_base + dst_start + patch_rows,
            ] = qkv_patches[src_patch, 2]
    return q_current


def q_current_patch_plan(
    group_index: int,
    q_heads_per_group: int,
    head_dim: int,
    patch_rows: int,
) -> tuple[tuple[int, int], ...]:
    if group_index < 0:
        raise ValueError("group_index must be non-negative")
    if head_dim % patch_rows != 0:
        raise ValueError("head_dim must be divisible by patch_rows")
    head_chunks = head_dim // patch_rows
    q_patches_per_group = q_heads_per_group * head_chunks
    entries: list[tuple[int, int]] = []
    for local_q_patch in range(q_patches_per_group):
        entries.append((0, group_index * q_patches_per_group + local_q_patch))
    for head_chunk in range(head_chunks):
        entries.append((1, group_index * head_chunks + head_chunk))
    for head_chunk in range(head_chunks):
        entries.append((2, group_index * head_chunks + head_chunk))
    return tuple(entries)
