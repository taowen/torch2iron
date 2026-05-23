# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy decode K/V into both present outputs and packet cache."""

import numpy as np

from aie.dialects.aiex import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.placers import SequentialPlacer


def _tap(buffer_size, sizes, strides, offset=0):
    sizes = [1] * (4 - len(sizes)) + list(sizes)
    strides = [0] * (4 - len(strides)) + list(strides)
    return TensorAccessPattern(
        tensor_dims=(int(buffer_size),),
        offset=int(offset),
        sizes=sizes,
        strides=strides,
    )


def _copy_once(rt, fifo_in, fifo_out, src, dst, src_tap, dst_tap):
    tg = rt.task_group()
    rt.fill(fifo_in.prod(), src, src_tap, task_group=tg)
    rt.drain(fifo_out.cons(), dst, dst_tap, task_group=tg, wait=True)
    rt.finish_task_group(tg)


def copy_present_packet_kv(
    dev,
    dtype,
    kv_dim,
    num_kv_groups,
    head_dim,
    packet_elements,
    packet_elements_per_group,
    key_packet_offset,
    value_packet_offset,
    transfer_size=None,
):
    dtype = np.dtype[dtype]
    if transfer_size is None:
        transfer_size = kv_dim

    key_ty = np.ndarray[(int(kv_dim),), dtype]
    value_ty = np.ndarray[(int(kv_dim),), dtype]
    present_ty = np.ndarray[(int(kv_dim),), dtype]
    packet_ty = np.ndarray[(int(packet_elements),), dtype]
    transfer_ty = np.ndarray[(int(transfer_size),), dtype]

    kv_input_tap = _tap(
        kv_dim,
        sizes=(num_kv_groups, head_dim),
        strides=(head_dim, 1),
    )
    present_tap = _tap(
        kv_dim,
        sizes=(num_kv_groups, head_dim),
        strides=(head_dim, 1),
    )
    key_packet_tap = _tap(
        packet_elements,
        sizes=(num_kv_groups, head_dim),
        strides=(packet_elements_per_group, 1),
        offset=key_packet_offset,
    )
    value_packet_tap = _tap(
        packet_elements,
        sizes=(num_kv_groups, head_dim),
        strides=(packet_elements_per_group, 1),
        offset=value_packet_offset,
    )

    key_present_in = ObjectFifo(transfer_ty, name="key_present_in", depth=1)
    key_present_out = key_present_in.cons().forward(
        name="key_present_out", depth=1
    )
    value_present_in = ObjectFifo(transfer_ty, name="value_present_in", depth=1)
    value_present_out = value_present_in.cons().forward(
        name="value_present_out", depth=1
    )
    key_packet_in = ObjectFifo(transfer_ty, name="key_packet_in", depth=1)
    key_packet_out = key_packet_in.cons().forward(name="key_packet_out", depth=1)
    value_packet_in = ObjectFifo(transfer_ty, name="value_packet_in", depth=1)
    value_packet_out = value_packet_in.cons().forward(
        name="value_packet_out", depth=1
    )

    rt = Runtime()
    with rt.sequence(key_ty, value_ty, present_ty, present_ty, packet_ty) as (
        key,
        value,
        present_key,
        present_value,
        packet,
    ):
        _copy_once(
            rt,
            key_present_in,
            key_present_out,
            key,
            present_key,
            kv_input_tap,
            present_tap,
        )
        _copy_once(
            rt,
            value_present_in,
            value_present_out,
            value,
            present_value,
            kv_input_tap,
            present_tap,
        )
        _copy_once(
            rt,
            key_packet_in,
            key_packet_out,
            key,
            packet,
            kv_input_tap,
            key_packet_tap,
        )
        _copy_once(
            rt,
            value_packet_in,
            value_packet_out,
            value,
            packet,
            kv_input_tap,
            value_packet_tap,
        )

    return Program(dev, rt).resolve_program(SequentialPlacer())
