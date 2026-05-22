#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aie.utils.hostruntime.xrtruntime.tensor import xrt as pyxrt

from models.fused_prefill.generated.decode_layout import (
    DECODE_PACKET_CACHE_NAMES,
    DECODE_PRESENT_KEY_NAMES,
    DECODE_PRESENT_VALUE_NAMES,
)
from models.fused_prefill.runtime_config import DECODE_ATTN_CHUNK_SIZE


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


def sync_decode_packet_range(packet_cache, start_element, num_elements):
    itemsize = packet_cache.dtype.itemsize
    sync_direction = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    packet_cache.buffer_object().sync(
        sync_direction,
        num_elements * itemsize,
        start_element * itemsize,
    )
    packet_cache.device = "npu"


def initialize_decode_packet_cache(
    config,
    aie_ops,
    max_seq_len,
    layer_idx,
    keys_cache,
    values_cache,
    num_preceding_tokens,
):
    packet_cache = aie_ops.decode.fused.get_buffer(DECODE_PACKET_CACHE_NAMES[layer_idx])
    packet = packet_cache.torch_view()
    packet.fill_(0)

    chunk_elements = decode_packet_chunk_elements(config)
    elements_per_group = decode_packet_elements_per_group(config, max_seq_len)
    num_chunks = max_seq_len // DECODE_ATTN_CHUNK_SIZE
    valid_tokens = min(num_preceding_tokens, max_seq_len)
    current_slot = aie_ops.decode.current_cache_slot

    for group_idx in range(config.n_kv_groups):
        group_base = group_idx * elements_per_group
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * DECODE_ATTN_CHUNK_SIZE
            chunk_end = chunk_start + DECODE_ATTN_CHUNK_SIZE
            chunk_base = group_base + chunk_idx * chunk_elements
            k_base = chunk_base
            v_base = chunk_base + DECODE_ATTN_CHUNK_SIZE * config.head_dim
            mask_base = chunk_base + 2 * DECODE_ATTN_CHUNK_SIZE * config.head_dim

            packet[
                k_base : k_base + DECODE_ATTN_CHUNK_SIZE * config.head_dim
            ] = keys_cache[group_idx, chunk_start:chunk_end, :].reshape(-1)
            packet[
                v_base : v_base + DECODE_ATTN_CHUNK_SIZE * config.head_dim
            ] = values_cache[group_idx, chunk_start:chunk_end, :].reshape(-1)

            valid_in_chunk = max(
                0, min(valid_tokens - chunk_start, DECODE_ATTN_CHUNK_SIZE)
            )
            if valid_in_chunk:
                packet[mask_base : mask_base + valid_in_chunk] = 1.0

        _, _, current_mask_offset = decode_packet_slot_offsets(
            config, max_seq_len, group_idx, current_slot
        )
        packet[current_mask_offset] = 1.0

    packet_cache.to("npu")


def sync_decode_packet_cache_slot(
    config,
    max_seq_len,
    packet_cache,
    present_key,
    present_value,
    dst_slot,
):
    packet = packet_cache.data
    for group_idx in range(config.n_kv_groups):
        k_offset, v_offset, mask_offset = decode_packet_slot_offsets(
            config, max_seq_len, group_idx, dst_slot
        )
        packet[k_offset : k_offset + config.head_dim] = present_key[group_idx]
        packet[v_offset : v_offset + config.head_dim] = present_value[group_idx]
        packet[mask_offset] = 1.0

        sync_decode_packet_range(packet_cache, k_offset, config.head_dim)
        sync_decode_packet_range(packet_cache, v_offset, config.head_dim)
        sync_decode_packet_range(packet_cache, mask_offset, 1)


def append_decode_kv_cache(config, aie_ops, max_seq_len, num_preceding_tokens):
    current_slot = aie_ops.decode.current_cache_slot
    dst_slot = num_preceding_tokens
    if dst_slot == current_slot:
        return

    for layer_idx in range(config.n_layers):
        present_key = (
            aie_ops.decode.fused.get_buffer(DECODE_PRESENT_KEY_NAMES[layer_idx])
            .data
            .reshape(config.n_kv_groups, config.head_dim)
        )
        present_value = (
            aie_ops.decode.fused.get_buffer(DECODE_PRESENT_VALUE_NAMES[layer_idx])
            .data
            .reshape(config.n_kv_groups, config.head_dim)
        )
        packet_cache = aie_ops.decode.fused.get_buffer(
            DECODE_PACKET_CACHE_NAMES[layer_idx]
        )
        sync_decode_packet_cache_slot(
            config,
            max_seq_len,
            packet_cache,
            present_key,
            present_value,
            dst_slot,
        )
