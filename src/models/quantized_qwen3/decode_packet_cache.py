#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aie.utils.hostruntime.xrtruntime.tensor import xrt as pyxrt

from models.quantized_qwen3.runtime_config import DECODE_ATTN_CHUNK_SIZE


def decode_packet_chunk_elements(config, chunk_size=DECODE_ATTN_CHUNK_SIZE):
    return 2 * chunk_size * config.head_dim + chunk_size


def decode_packet_elements_per_group(
    config,
    max_seq_len,
    chunk_size=DECODE_ATTN_CHUNK_SIZE,
):
    return (max_seq_len // chunk_size) * decode_packet_chunk_elements(
        config, chunk_size
    )


def decode_packet_slot_offsets(
    config,
    max_seq_len,
    group_idx,
    slot,
    chunk_size=DECODE_ATTN_CHUNK_SIZE,
):
    chunk_idx = slot // chunk_size
    row = slot % chunk_size
    chunk_elements = decode_packet_chunk_elements(config, chunk_size)
    group_base = group_idx * decode_packet_elements_per_group(
        config, max_seq_len, chunk_size
    )
    chunk_base = group_base + chunk_idx * chunk_elements
    k_offset = chunk_base + row * config.head_dim
    v_offset = chunk_base + chunk_size * config.head_dim + row * config.head_dim
    mask_offset = chunk_base + 2 * chunk_size * config.head_dim + row
    return k_offset, v_offset, mask_offset


def decode_packet_chunk_range(
    config,
    max_seq_len,
    group_idx,
    slot,
    chunk_size=DECODE_ATTN_CHUNK_SIZE,
):
    chunk_idx = slot // chunk_size
    chunk_elements = decode_packet_chunk_elements(config, chunk_size)
    group_base = group_idx * decode_packet_elements_per_group(
        config, max_seq_len, chunk_size
    )
    chunk_start = group_base + chunk_idx * chunk_elements
    return chunk_start, chunk_elements


def sync_decode_packet_range(packet_cache, start_element, num_elements):
    itemsize = packet_cache.dtype.itemsize
    sync_direction = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    packet_cache.buffer_object().sync(
        sync_direction,
        num_elements * itemsize,
        start_element * itemsize,
    )
    packet_cache.device = "npu"


def sync_decode_packet_ranges(packet_cache, ranges):
    merged_ranges = []
    for start, length in sorted(ranges):
        if length <= 0:
            continue
        end = start + length
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    for start, end in merged_ranges:
        sync_decode_packet_range(packet_cache, start, end - start)


def sync_decode_packet_cache_slot(
    config,
    max_seq_len,
    packet_cache,
    present_key,
    present_value,
    dst_slot,
):
    packet = packet_cache.data
    touched_chunks = []
    for group_idx in range(config.n_kv_groups):
        k_offset, v_offset, mask_offset = decode_packet_slot_offsets(
            config, max_seq_len, group_idx, dst_slot
        )
        packet[k_offset : k_offset + config.head_dim] = present_key[group_idx]
        packet[v_offset : v_offset + config.head_dim] = present_value[group_idx]
        packet[mask_offset] = 1.0
        touched_chunks.append(
            decode_packet_chunk_range(config, max_seq_len, group_idx, dst_slot)
        )

    sync_decode_packet_ranges(packet_cache, touched_chunks)
