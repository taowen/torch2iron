# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
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


def qwen_chunked_attention_current(
    dev,
    packet_seq_len,
    attend_seq_len,
    current_slot,
    num_kv_groups,
    q_heads_per_group,
    head_dim,
    chunk_size,
    kernel_object="qwen_chunked_attention_current.o",
    verbose=False,
    func_prefix="",
    trace_size=0,
    trace_ddr_id=5,
    packed_fifo_depth=1,
):
    if packet_seq_len % chunk_size != 0:
        raise ValueError("packet_seq_len must be divisible by chunk_size")
    if attend_seq_len % chunk_size != 0:
        raise ValueError("attend_seq_len must be divisible by chunk_size")
    if attend_seq_len > packet_seq_len:
        raise ValueError("attend_seq_len must be <= packet_seq_len")
    if current_slot < 0 or current_slot >= attend_seq_len:
        raise ValueError("current_slot must be inside attend_seq_len")
    if packed_fifo_depth <= 0:
        raise ValueError("packed_fifo_depth must be positive")
    if verbose:
        print(
            "qwen_chunked_attention_current: "
            f"num_kv_groups={num_kv_groups}, q_heads_per_group={q_heads_per_group}, "
            f"head_dim={head_dim}, packet_seq_len={packet_seq_len}, "
            f"attend_seq_len={attend_seq_len}, current_slot={current_slot}, "
            f"chunk_size={chunk_size}"
        )

    num_chunks = attend_seq_len // chunk_size
    q_elements_per_group = q_heads_per_group * head_dim
    q_elements = num_kv_groups * q_elements_per_group
    q_current_elements_per_group = q_elements_per_group + 2 * head_dim
    q_current_elements = num_kv_groups * q_current_elements_per_group
    packed_chunk_elements = 2 * chunk_size * head_dim + chunk_size
    packet_elements_per_group = packet_seq_len // chunk_size * packed_chunk_elements
    active_packet_elements_per_group = num_chunks * packed_chunk_elements
    packet_elements = num_kv_groups * packet_elements_per_group
    current_chunk = current_slot // chunk_size
    current_row = current_slot % chunk_size
    dtype = bfloat16
    kernel_object = f"{func_prefix}{kernel_object}"

    q_current_l3_ty = np.ndarray[(q_current_elements,), np.dtype[dtype]]
    packet_l3_ty = np.ndarray[(packet_elements,), np.dtype[dtype]]
    out_l3_ty = np.ndarray[(q_elements,), np.dtype[dtype]]

    q_current_group_ty = np.ndarray[(q_current_elements_per_group,), np.dtype[dtype]]
    packed_chunk_ty = np.ndarray[(packed_chunk_elements,), np.dtype[dtype]]
    out_group_ty = np.ndarray[(q_elements_per_group,), np.dtype[dtype]]
    state_ty = np.ndarray[(q_heads_per_group * 2,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(q_elements_per_group,), np.dtype[np.float32]]

    init_kernel = Kernel(
        f"{func_prefix}llama_chunked_attention_init_f32",
        kernel_object,
        [state_ty, acc_ty, np.int32, np.int32],
    )
    update_kernel = Kernel(
        f"{func_prefix}llama_chunked_attention_update_packed_bf16",
        kernel_object,
        [
            q_current_group_ty,
            packed_chunk_ty,
            state_ty,
            acc_ty,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    update_current_kernel = Kernel(
        f"{func_prefix}qwen_chunked_attention_current_direct_bf16",
        kernel_object,
        [
            q_current_group_ty,
            packed_chunk_ty,
            state_ty,
            acc_ty,
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
    q_fifos = [
        ObjectFifo(q_current_group_ty, name=f"qwen_attn_q_current_g{group}", depth=1)
        for group in range(num_kv_groups)
    ]
    packed_fifos = [
        ObjectFifo(
            packed_chunk_ty,
            name=f"qwen_attn_packed_chunks_g{group}",
            depth=packed_fifo_depth,
        )
        for group in range(num_kv_groups)
    ]
    out_fifos = [
        ObjectFifo(out_group_ty, name=f"qwen_attn_context_g{group}", depth=1)
        for group in range(num_kv_groups)
    ]
    states = [
        Buffer(
            initial_value=np.zeros(shape=(q_heads_per_group * 2,), dtype=np.float32),
            name=f"qwen_attn_state_g{group}",
        )
        for group in range(num_kv_groups)
    ]
    accs = [
        Buffer(
            initial_value=np.zeros(shape=(q_elements_per_group,), dtype=np.float32),
            name=f"qwen_attn_acc_g{group}",
        )
        for group in range(num_kv_groups)
    ]

    def worker_body(
        q_fifo,
        packed_fifo,
        out_fifo,
        state,
        acc,
        init_kernel,
        update_kernel,
        update_current_kernel,
        finalize_kernel,
    ):
        q_current = q_fifo.acquire(1)
        out = out_fifo.acquire(1)
        init_kernel(state, acc, q_heads_per_group, head_dim)
        for _ in range_(current_chunk):
            packed = packed_fifo.acquire(1)
            update_kernel(
                q_current,
                packed,
                state,
                acc,
                q_heads_per_group,
                chunk_size,
                head_dim,
            )
            packed_fifo.release(1)
        packed = packed_fifo.acquire(1)
        update_current_kernel(
            q_current,
            packed,
            state,
            acc,
            current_row,
            q_heads_per_group,
            chunk_size,
            head_dim,
        )
        packed_fifo.release(1)
        for _ in range_(num_chunks - current_chunk - 1):
            packed = packed_fifo.acquire(1)
            update_kernel(
                q_current,
                packed,
                state,
                acc,
                q_heads_per_group,
                chunk_size,
                head_dim,
            )
            packed_fifo.release(1)
        finalize_kernel(state, acc, out, q_heads_per_group, head_dim)
        out_fifo.release(1)
        q_fifo.release(1)

    workers = [
        Worker(
            worker_body,
            [
                q_fifos[group].cons(),
                packed_fifos[group].cons(),
                out_fifos[group].prod(),
                states[group],
                accs[group],
                init_kernel,
                update_kernel,
                update_current_kernel,
                finalize_kernel,
            ],
            stack_size=0xD00,
            placement=Tile(group, 2),
            trace=1 if trace_size > 0 and group == 0 else None,
        )
        for group in range(num_kv_groups)
    ]

    q_current_taps = [
        _tap(
            q_current_elements,
            group * q_current_elements_per_group,
            q_current_elements_per_group,
        )
        for group in range(num_kv_groups)
    ]
    packet_active_taps = [
        _tap(
            packet_elements,
            group * packet_elements_per_group,
            active_packet_elements_per_group,
        )
        for group in range(num_kv_groups)
    ]
    sequence_types = [q_current_l3_ty, packet_l3_ty, out_l3_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        q_current_l3, packet_l3, out_l3 = runtime_args[:3]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[workers[0]], ddr_id=trace_ddr_id)

        rt.start(*workers)
        attn_tg = rt.task_group()
        for group in range(num_kv_groups):
            rt.fill(
                q_fifos[group].prod(),
                q_current_l3,
                q_current_taps[group],
                task_group=attn_tg,
            )
            rt.fill(
                packed_fifos[group].prod(),
                packet_l3,
                packet_active_taps[group],
                task_group=attn_tg,
            )
        for group in range(num_kv_groups):
            rt.drain(
                out_fifos[group].cons(),
                out_l3,
                _tap(q_elements, group * q_elements_per_group, q_elements_per_group),
                wait=True,
                task_group=attn_tg,
            )
        rt.finish_task_group(attn_tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
