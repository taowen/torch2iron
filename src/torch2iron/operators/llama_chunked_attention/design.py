# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def llama_chunked_attention(
    dev,
    max_seq_len,
    num_kv_groups,
    q_heads_per_group,
    head_dim,
    chunk_size,
    kernel_object="llama_chunked_attention.o",
    verbose=False,
    func_prefix="",
):
    if max_seq_len % chunk_size != 0:
        raise ValueError("max_seq_len must be divisible by chunk_size")
    if num_kv_groups <= 0:
        raise ValueError("num_kv_groups must be positive")
    if q_heads_per_group <= 0:
        raise ValueError("q_heads_per_group must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if verbose:
        print(
            "llama_chunked_attention: "
            f"num_kv_groups={num_kv_groups}, "
            f"q_heads_per_group={q_heads_per_group}, head_dim={head_dim}, "
            f"max_seq_len={max_seq_len}, chunk_size={chunk_size}"
        )

    num_chunks = max_seq_len // chunk_size
    q_elements_per_group = q_heads_per_group * head_dim
    q_elements = num_kv_groups * q_elements_per_group
    packed_chunk_elements = 2 * chunk_size * head_dim + chunk_size
    packed_elements_per_group = num_chunks * packed_chunk_elements
    packed_elements = num_kv_groups * packed_elements_per_group
    dtype = bfloat16
    kernel_object = f"{func_prefix}{kernel_object}"

    q_l3_ty = np.ndarray[(q_elements,), np.dtype[dtype]]
    packed_l3_ty = np.ndarray[(packed_elements,), np.dtype[dtype]]
    out_l3_ty = np.ndarray[(q_elements,), np.dtype[dtype]]

    q_group_ty = np.ndarray[(q_elements_per_group,), np.dtype[dtype]]
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
            q_group_ty,
            packed_chunk_ty,
            state_ty,
            acc_ty,
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
        ObjectFifo(q_group_ty, name=f"llama_attn_q_g{group}", depth=1)
        for group in range(num_kv_groups)
    ]
    packed_fifos = [
        ObjectFifo(
            packed_chunk_ty,
            name=f"llama_attn_packed_kv_chunks_g{group}",
            depth=1,
        )
        for group in range(num_kv_groups)
    ]
    out_fifos = [
        ObjectFifo(out_group_ty, name=f"llama_attn_context_g{group}", depth=1)
        for group in range(num_kv_groups)
    ]

    states = [
        Buffer(
            initial_value=np.zeros(shape=(q_heads_per_group * 2,), dtype=np.float32),
            name=f"llama_attn_state_g{group}",
        )
        for group in range(num_kv_groups)
    ]
    accs = [
        Buffer(
            initial_value=np.zeros(shape=(q_elements_per_group,), dtype=np.float32),
            name=f"llama_attn_acc_g{group}",
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
        finalize_kernel,
    ):
        q = q_fifo.acquire(1)
        out = out_fifo.acquire(1)
        init_kernel(state, acc, q_heads_per_group, head_dim)
        for _ in range_(num_chunks):
            packed = packed_fifo.acquire(1)
            update_kernel(
                q,
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
                finalize_kernel,
            ],
            stack_size=0xD00,
        )
        for group in range(num_kv_groups)
    ]

    q_taps = [
        TensorAccessPattern(
            (q_elements,),
            group * q_elements_per_group,
            [1, 1, 1, q_elements_per_group],
            [0, 0, 0, 1],
        )
        for group in range(num_kv_groups)
    ]
    packed_taps = [
        TensorAccessPattern(
            (packed_elements,),
            group * packed_elements_per_group,
            [1, 1, 1, packed_elements_per_group],
            [0, 0, 0, 1],
        )
        for group in range(num_kv_groups)
    ]

    rt = Runtime()
    with rt.sequence(q_l3_ty, packed_l3_ty, out_l3_ty) as (q_l3, packed_l3, out_l3):
        rt.start(*workers)
        tg = rt.task_group()
        for group in range(num_kv_groups):
            rt.fill(q_fifos[group].prod(), q_l3, q_taps[group], task_group=tg)
            rt.fill(
                packed_fifos[group].prod(),
                packed_l3,
                packed_taps[group],
                task_group=tg,
            )
        for group in range(num_kv_groups):
            rt.drain(
                out_fifos[group].cons(),
                out_l3,
                q_taps[group],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
