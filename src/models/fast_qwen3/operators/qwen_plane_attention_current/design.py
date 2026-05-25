# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer


def _tap(buffer_elements, offset, length):
    return TensorAccessPattern(
        (buffer_elements,),
        offset,
        [1, 1, 1, length],
        [0, 0, 0, 1],
    )


def _plane_pair_grouped_tap(packet_seq_len, head_dim, plane_pair_base, chunk, tile_size):
    token_stride_elements = 4 * head_dim
    plane_elements = packet_seq_len * token_stride_elements
    return TensorAccessPattern(
        (4, 4, packet_seq_len, head_dim),
        plane_pair_base * plane_elements + chunk * tile_size * token_stride_elements,
        [4, 2, tile_size, head_dim],
        [head_dim, plane_elements, token_stride_elements, 1],
    )


def qwen_plane_attention_current(
    dev,
    packet_seq_len,
    attend_seq_len,
    current_slot,
    q_heads_per_group,
    head_dim,
    tile_size,
    kernel_object="qwen_plane_attention_current.o",
    verbose=False,
    func_prefix="",
    plane_fifo_depth=2,
):
    if attend_seq_len > packet_seq_len:
        raise ValueError("attend_seq_len must be <= packet_seq_len")
    num_chunks = (attend_seq_len + tile_size - 1) // tile_size
    if num_chunks * tile_size > packet_seq_len:
        raise ValueError("rounded attend_seq_len must fit inside packet_seq_len")
    if current_slot < 0 or current_slot >= attend_seq_len:
        raise ValueError("current_slot must be inside attend_seq_len")
    if plane_fifo_depth <= 0:
        raise ValueError("plane_fifo_depth must be positive")
    if verbose:
        print(
            "qwen_plane_attention_current: "
            f"q_heads_per_group={q_heads_per_group}, head_dim={head_dim}, "
            f"packet_seq_len={packet_seq_len}, attend_seq_len={attend_seq_len}, "
            f"current_slot={current_slot}, tile_size={tile_size}"
        )

    num_kv_groups = 8
    plane_group_count = 4
    plane_pair_count = 2
    q_elements_per_group = q_heads_per_group * head_dim
    q_current_elements_per_group = q_elements_per_group + 2 * head_dim
    q_current_elements = num_kv_groups * q_current_elements_per_group
    q_elements = num_kv_groups * q_elements_per_group
    q_pair_elements = plane_group_count * q_current_elements_per_group
    token_stride_elements = plane_group_count * head_dim
    plane_elements = packet_seq_len * token_stride_elements
    kv_plane_elements = 4 * plane_elements
    plane_chunk_elements = tile_size * token_stride_elements
    plane_pair_chunk_elements = 2 * plane_chunk_elements
    plane_group_pair_chunk_elements = 2 * tile_size * head_dim
    out_pair_elements = plane_group_count * q_elements_per_group
    current_chunk = current_slot // tile_size
    current_row = current_slot % tile_size
    dtype = bfloat16
    kernel_object = f"{func_prefix}{kernel_object}"

    q_current_l3_ty = np.ndarray[(q_current_elements,), np.dtype[dtype]]
    kv_plane_l3_ty = np.ndarray[(kv_plane_elements,), np.dtype[dtype]]
    out_l3_ty = np.ndarray[(q_elements,), np.dtype[dtype]]

    q_pair_ty = np.ndarray[(q_pair_elements,), np.dtype[dtype]]
    q_current_group_ty = np.ndarray[(q_current_elements_per_group,), np.dtype[dtype]]
    plane_pair_chunk_ty = np.ndarray[(plane_pair_chunk_elements,), np.dtype[dtype]]
    plane_group_pair_chunk_ty = np.ndarray[
        (plane_group_pair_chunk_elements,), np.dtype[dtype]
    ]
    out_group_ty = np.ndarray[(q_elements_per_group,), np.dtype[dtype]]
    out_pair_ty = np.ndarray[(out_pair_elements,), np.dtype[dtype]]
    state_ty = np.ndarray[(q_heads_per_group * 2,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(q_elements_per_group,), np.dtype[np.float32]]

    init_kernel = Kernel(
        f"{func_prefix}llama_chunked_attention_init_f32",
        kernel_object,
        [state_ty, acc_ty, np.int32, np.int32],
    )
    update_kernel = Kernel(
        f"{func_prefix}qwen_plane_group_attention_update_bf16",
        kernel_object,
        [
            q_current_group_ty,
            plane_group_pair_chunk_ty,
            state_ty,
            acc_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    finalize_kernel = Kernel(
        f"{func_prefix}llama_chunked_attention_finalize_bf16",
        kernel_object,
        [state_ty, acc_ty, out_group_ty, np.int32, np.int32],
    )

    q_pair_fifos = [
        ObjectFifo(q_pair_ty, name=f"qwen_plane_attn_q_p{pair}", depth=1)
        for pair in range(plane_pair_count)
    ]
    q_fifos_by_pair = [
        q_pair_fifos[pair]
        .cons()
        .split(
            offsets=[
                group * q_current_elements_per_group
                for group in range(plane_group_count)
            ],
            obj_types=[q_current_group_ty] * plane_group_count,
            names=[
                f"qwen_plane_attn_q_g{pair * plane_group_count + group}"
                for group in range(plane_group_count)
            ],
            depths=[1] * plane_group_count,
            placement=Tile(pair * plane_group_count + 1, 1),
        )
        for pair in range(plane_pair_count)
    ]
    q_fifos = [
        q_fifos_by_pair[group // plane_group_count][group % plane_group_count]
        for group in range(num_kv_groups)
    ]
    plane_reader_fifos = [
        ObjectFifo(
            plane_pair_chunk_ty,
            name=f"qwen_plane_attn_kv_pair_p{pair}",
            depth=plane_fifo_depth,
        )
        for pair in range(plane_pair_count)
    ]
    plane_fifos_by_pair = [
        plane_reader_fifos[pair]
        .cons()
        .split(
            offsets=[
                group * plane_group_pair_chunk_elements
                for group in range(plane_group_count)
            ],
            obj_types=[plane_group_pair_chunk_ty] * plane_group_count,
            names=[
                f"qwen_plane_attn_kv_pair_mem_g{pair * plane_group_count + group}"
                for group in range(plane_group_count)
            ],
            depths=[plane_fifo_depth] * plane_group_count,
            placement=Tile(pair * plane_group_count, 1),
        )
        for pair in range(plane_pair_count)
    ]
    plane_fifos = [
        plane_fifos_by_pair[group // plane_group_count][group % plane_group_count]
        for group in range(num_kv_groups)
    ]
    out_pair_fifos = [
        ObjectFifo(
            out_pair_ty,
            name=f"qwen_plane_attn_context_p{pair}",
            depth=1,
        )
        for pair in range(plane_pair_count)
    ]
    out_fifos_by_pair = [
        out_pair_fifos[pair]
        .prod()
        .join(
            offsets=[
                group * q_elements_per_group for group in range(plane_group_count)
            ],
            obj_types=[out_group_ty] * plane_group_count,
            names=[
                f"qwen_plane_attn_context_g{pair * plane_group_count + group}"
                for group in range(plane_group_count)
            ],
            depths=[1] * plane_group_count,
            placement=Tile(pair * plane_group_count, 1),
        )
        for pair in range(plane_pair_count)
    ]
    out_fifos = [
        out_fifos_by_pair[group // plane_group_count][group % plane_group_count]
        for group in range(num_kv_groups)
    ]
    states = [
        Buffer(
            initial_value=np.zeros(shape=(q_heads_per_group * 2,), dtype=np.float32),
            name=f"qwen_plane_attn_state_g{group}",
        )
        for group in range(num_kv_groups)
    ]
    accs = [
        Buffer(
            initial_value=np.zeros(shape=(q_elements_per_group,), dtype=np.float32),
            name=f"qwen_plane_attn_acc_g{group}",
        )
        for group in range(num_kv_groups)
    ]

    def worker_body(
        q_fifo,
        plane_fifo,
        out_fifo,
        state,
        acc,
        init_kernel,
        update_kernel,
        finalize_kernel,
    ):
        q_current = q_fifo.acquire(1)
        out = out_fifo.acquire(1)
        init_kernel(state, acc, q_heads_per_group, head_dim)
        for chunk in range(num_chunks):
            plane_pair = plane_fifo.acquire(1)
            row = current_row if chunk == current_chunk else -1
            valid_len = min(tile_size, attend_seq_len - chunk * tile_size)
            update_kernel(
                q_current,
                plane_pair,
                state,
                acc,
                row,
                valid_len,
                q_heads_per_group,
                tile_size,
                head_dim,
            )
            plane_fifo.release(1)
        finalize_kernel(state, acc, out, q_heads_per_group, head_dim)
        out_fifo.release(1)
        q_fifo.release(1)

    workers = [
        Worker(
            worker_body,
            [
                q_fifos[group].cons(),
                plane_fifos[group].cons(),
                out_fifos[group].prod(),
                states[group],
                accs[group],
                init_kernel,
                update_kernel,
                finalize_kernel,
            ],
            stack_size=0xD00,
            placement=Tile(group, 2),
        )
        for group in range(num_kv_groups)
    ]

    q_current_taps = [
        _tap(q_current_elements, pair * q_pair_elements, q_pair_elements)
        for pair in range(plane_pair_count)
    ]
    plane_pair_taps = []
    for pair in range(plane_pair_count):
        plane_pair_base = pair * plane_pair_count
        group_taps = []
        for chunk in range(num_chunks):
            group_taps.append(
                _plane_pair_grouped_tap(
                    packet_seq_len,
                    head_dim,
                    plane_pair_base,
                    chunk,
                    tile_size,
                )
            )
        plane_pair_taps.append(group_taps)

    rt = Runtime()
    with rt.sequence(q_current_l3_ty, kv_plane_l3_ty, out_l3_ty) as runtime_args:
        q_current_l3, kv_plane_l3, out_l3 = runtime_args[:3]
        rt.start(*workers)
        attn_tg = rt.task_group()
        for pair in range(plane_pair_count):
            rt.fill(
                q_pair_fifos[pair].prod(),
                q_current_l3,
                q_current_taps[pair],
                task_group=attn_tg,
            )
        for pair in range(plane_pair_count):
            for chunk in range(num_chunks):
                rt.fill(
                    plane_reader_fifos[pair].prod(),
                    kv_plane_l3,
                    plane_pair_taps[pair][chunk],
                    task_group=attn_tg,
                )
        for pair in range(plane_pair_count):
            rt.drain(
                out_pair_fifos[pair].cons(),
                out_l3,
                _tap(q_elements, pair * out_pair_elements, out_pair_elements),
                wait=True,
                task_group=attn_tg,
            )
        rt.finish_task_group(attn_tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
