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


def qwen_current_kv_plane_write(
    dev,
    packet_seq_len,
    current_slot,
    q_heads_per_group,
    head_dim,
    verbose=False,
):
    if current_slot < 0 or current_slot >= packet_seq_len:
        raise ValueError("current_slot must be inside packet_seq_len")
    if verbose:
        print(
            "qwen_current_kv_plane_write: "
            f"packet_seq_len={packet_seq_len}, current_slot={current_slot}, "
            f"q_heads_per_group={q_heads_per_group}, head_dim={head_dim}"
        )

    num_kv_groups = 8
    plane_group_count = 4
    q_elements_per_group = q_heads_per_group * head_dim
    q_current_elements_per_group = q_elements_per_group + 2 * head_dim
    q_current_elements = num_kv_groups * q_current_elements_per_group
    token_stride_elements = plane_group_count * head_dim
    plane_elements = packet_seq_len * token_stride_elements
    kv_plane_elements = 4 * plane_elements
    dtype = bfloat16

    q_current_ty = np.ndarray[(q_current_elements,), np.dtype[dtype]]
    kv_plane_ty = np.ndarray[(kv_plane_elements,), np.dtype[dtype]]
    kv_row_ty = np.ndarray[(head_dim,), np.dtype[dtype]]

    kv_fifo = ObjectFifo(kv_row_ty, name="qwen_current_kv_plane_write_row", depth=1)
    kv_out = kv_fifo.cons().forward(
        name="qwen_current_kv_plane_write_row_out",
        depth=1,
    )

    sequence_types = [q_current_ty, kv_plane_ty]
    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        q_current_l3, kv_plane_l3 = runtime_args[:2]
        for group in range(num_kv_groups):
            q_current_group_offset = group * q_current_elements_per_group
            key_src_offset = q_current_group_offset + q_elements_per_group
            value_src_offset = key_src_offset + head_dim
            token_base = current_slot * token_stride_elements
            if group < plane_group_count:
                key_plane = 0
                value_plane = 1
                group_in_plane = group
            else:
                key_plane = 2
                value_plane = 3
                group_in_plane = group - plane_group_count
            row_offset = token_base + group_in_plane * head_dim
            key_dst_offset = key_plane * plane_elements + row_offset
            value_dst_offset = value_plane * plane_elements + row_offset

            _copy_once(
                rt,
                kv_fifo,
                kv_out,
                q_current_l3,
                kv_plane_l3,
                _tap(q_current_elements, key_src_offset, head_dim),
                _tap(kv_plane_elements, key_dst_offset, head_dim),
            )
            _copy_once(
                rt,
                kv_fifo,
                kv_out,
                q_current_l3,
                kv_plane_l3,
                _tap(q_current_elements, value_src_offset, head_dim),
                _tap(kv_plane_elements, value_dst_offset, head_dim),
            )

    return Program(dev, rt).resolve_program(SequentialPlacer())
