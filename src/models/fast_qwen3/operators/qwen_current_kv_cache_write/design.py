# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.placers import SequentialPlacer


def _tap(buffer_elements, offset, length):
    return TensorAccessPattern(
        (buffer_elements,),
        offset,
        [1, 1, 1, length],
        [0, 0, 0, 1],
    )


def _copy_once(rt, fifo_in, fifo_out, src, dst, src_tap, dst_tap):
    tg = rt.task_group()
    rt.fill(fifo_in.prod(), src, src_tap, task_group=tg)
    rt.drain(fifo_out.cons(), dst, dst_tap, wait=True, task_group=tg)
    rt.finish_task_group(tg)


def qwen_current_kv_cache_write(
    dev,
    packet_seq_len,
    current_slot,
    num_kv_groups,
    q_heads_per_group,
    head_dim,
    chunk_size,
    verbose=False,
):
    if packet_seq_len % chunk_size != 0:
        raise ValueError("packet_seq_len must be divisible by chunk_size")
    if current_slot < 0 or current_slot >= packet_seq_len:
        raise ValueError("current_slot must be inside packet_seq_len")
    if verbose:
        print(
            "qwen_current_kv_cache_write: "
            f"num_kv_groups={num_kv_groups}, q_heads_per_group={q_heads_per_group}, "
            f"head_dim={head_dim}, packet_seq_len={packet_seq_len}, "
            f"current_slot={current_slot}, chunk_size={chunk_size}"
        )

    q_elements_per_group = q_heads_per_group * head_dim
    q_current_elements_per_group = q_elements_per_group + 2 * head_dim
    q_current_elements = num_kv_groups * q_current_elements_per_group
    chunk_elements = 2 * chunk_size * head_dim + chunk_size
    packet_elements_per_group = packet_seq_len // chunk_size * chunk_elements
    packet_elements = num_kv_groups * packet_elements_per_group
    current_chunk = current_slot // chunk_size
    current_row = current_slot % chunk_size
    mask_write_row = current_row if current_row == 0 else current_row - 1
    key_packet_offset = current_chunk * chunk_elements + current_row * head_dim
    value_packet_offset = (
        current_chunk * chunk_elements + chunk_size * head_dim + current_row * head_dim
    )
    mask_packet_offset = (
        current_chunk * chunk_elements + 2 * chunk_size * head_dim + mask_write_row
    )
    dtype = bfloat16

    q_current_ty = np.ndarray[(q_current_elements,), np.dtype[dtype]]
    mask_ty = np.ndarray[(2,), np.dtype[dtype]]
    packet_ty = np.ndarray[(packet_elements,), np.dtype[dtype]]
    kv_row_ty = np.ndarray[(head_dim,), np.dtype[dtype]]

    kv_fifo = ObjectFifo(kv_row_ty, name="qwen_current_kv_cache_write_row", depth=1)
    kv_out = kv_fifo.cons().forward(
        name="qwen_current_kv_cache_write_row_out",
        depth=1,
    )
    mask_fifo = ObjectFifo(mask_ty, name="qwen_current_kv_cache_write_mask", depth=1)
    mask_out = mask_fifo.cons().forward(
        name="qwen_current_kv_cache_write_mask_out",
        depth=1,
    )

    sequence_types = [q_current_ty, mask_ty, packet_ty]
    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        q_current_l3, mask_l3, packet_l3 = runtime_args[:3]
        for group in range(num_kv_groups):
            q_current_group_offset = group * q_current_elements_per_group
            key_src_offset = q_current_group_offset + q_elements_per_group
            value_src_offset = key_src_offset + head_dim
            packet_group_offset = group * packet_elements_per_group
            _copy_once(
                rt,
                kv_fifo,
                kv_out,
                q_current_l3,
                packet_l3,
                _tap(q_current_elements, key_src_offset, head_dim),
                _tap(packet_elements, packet_group_offset + key_packet_offset, head_dim),
            )
            _copy_once(
                rt,
                kv_fifo,
                kv_out,
                q_current_l3,
                packet_l3,
                _tap(q_current_elements, value_src_offset, head_dim),
                _tap(packet_elements, packet_group_offset + value_packet_offset, head_dim),
            )
            _copy_once(
                rt,
                mask_fifo,
                mask_out,
                mask_l3,
                packet_l3,
                _tap(2, 0, 2),
                _tap(packet_elements, packet_group_offset + mask_packet_offset, 2),
            )

    return Program(dev, rt).resolve_program(SequentialPlacer())
