# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer


PATCH_ROWS = 64


def q4nx_fused_q_current_projection(
    dev,
    in_features,
    num_kv_groups,
    group_index,
    q_heads_per_group,
    head_dim,
    trace_size=0,
    trace_ddr_id=7,
    func_prefix="",
    kernel_object="q4nx_fused_q_current_projection.o",
    verbose=False,
):
    if in_features % 256 != 0:
        raise ValueError("in_features must be divisible by 256")
    if num_kv_groups <= 0:
        raise ValueError("num_kv_groups must be positive")
    if head_dim % PATCH_ROWS != 0:
        raise ValueError("head_dim must be divisible by 64")

    chunk_bytes = 5120
    patch_bytes_per_k_chunk = 2 * chunk_bytes
    k_chunks = in_features // 256
    head_chunks = head_dim // PATCH_ROWS
    q_patches_per_group = q_heads_per_group * head_chunks
    q_current_patches = q_patches_per_group + 2 * head_chunks
    if q_current_patches > 8:
        raise ValueError("q_current projection currently supports at most 8 patches")
    if verbose:
        print(
            "Q4NX fused q_current projection: "
            f"K={in_features}, num_kv_groups={num_kv_groups}, "
            f"group_index={group_index}, "
            f"q_heads_per_group={q_heads_per_group}, head_dim={head_dim}"
        )

    weight_stream_bytes = (
        num_kv_groups * q_current_patches * k_chunks * patch_bytes_per_k_chunk
    )
    q_elements_per_group = q_heads_per_group * head_dim
    q_current_elements_per_group = q_elements_per_group + 2 * head_dim
    q_current_elements = num_kv_groups * q_current_elements_per_group

    hidden_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    norm_weight_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    normed_hidden_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    weight_stream_ty = np.ndarray[(weight_stream_bytes,), np.dtype[np.uint8]]
    weight_chunk_ty = np.ndarray[(patch_bytes_per_k_chunk,), np.dtype[np.uint8]]
    patch_out_ty = np.ndarray[(PATCH_ROWS,), np.dtype[bfloat16]]
    q_current_ty = np.ndarray[(q_current_elements,), np.dtype[bfloat16]]

    hidden_fifo = ObjectFifo(hidden_ty, name="q4nx_qcur_hidden", depth=1)
    norm_weight_fifo = ObjectFifo(
        norm_weight_ty,
        name="q4nx_qcur_norm_weight",
        depth=1,
    )
    normed_hidden_l1l2 = ObjectFifo(
        normed_hidden_ty,
        name="q4nx_qcur_normed_hidden_l1l2",
        depth=1,
    )
    normed_hidden_l2l1 = normed_hidden_l1l2.cons().forward(
        obj_type=normed_hidden_ty,
        name="q4nx_qcur_normed_hidden_l2l1",
        depth=1,
        placement=Tile(0, 1),
    )
    weight_fifos = [
        ObjectFifo(weight_chunk_ty, name=f"q4nx_qcur_weight_{patch_idx}", depth=1)
        for patch_idx in range(q_current_patches)
    ]
    out_fifos = [
        ObjectFifo(patch_out_ty, name=f"q4nx_qcur_out_{patch_idx}", depth=1)
        for patch_idx in range(q_current_patches)
    ]

    norm_kernel = Kernel(
        f"{func_prefix}q4nx_rms_norm_full",
        f"{func_prefix}{kernel_object}",
        [hidden_ty, norm_weight_ty, normed_hidden_ty],
    )
    projection_kernel = Kernel(
        f"{func_prefix}q4nx_fused_projection_patch",
        f"{func_prefix}{kernel_object}",
        [
            np.int32,
            np.int32,
            normed_hidden_ty,
            weight_chunk_ty,
            patch_out_ty,
        ],
    )

    def norm_body(hidden_fifo, norm_weight_fifo, normed_hidden_fifo, norm_kernel):
        hidden = hidden_fifo.acquire(1)
        norm_weight = norm_weight_fifo.acquire(1)
        normed_hidden = normed_hidden_fifo.acquire(1)
        norm_kernel(hidden, norm_weight, normed_hidden)
        normed_hidden_fifo.release(1)
        norm_weight_fifo.release(1)
        hidden_fifo.release(1)

    def core_body(normed_hidden_fifo, weight_fifo, out_fifo, projection_kernel):
        normed_hidden = normed_hidden_fifo.acquire(1)
        for _group_pos in range(num_kv_groups):
            out = out_fifo.acquire(1)
            for k_idx in range(k_chunks):
                weight = weight_fifo.acquire(1)
                projection_kernel(
                    1 if k_idx == 0 else 0,
                    k_idx * 256,
                    normed_hidden,
                    weight,
                    out,
                )
                weight_fifo.release(1)
            out_fifo.release(1)
        normed_hidden_fifo.release(1)

    norm_worker = Worker(
        norm_body,
        [
            hidden_fifo.cons(),
            norm_weight_fifo.cons(),
            normed_hidden_l1l2.prod(),
            norm_kernel,
        ],
        placement=Tile(0, 3),
    )
    projection_workers = [
        Worker(
            core_body,
            [
                normed_hidden_l2l1.cons(),
                weight_fifos[patch_idx].cons(),
                out_fifos[patch_idx].prod(),
                projection_kernel,
            ],
            placement=Tile(patch_idx, 2),
            trace=1 if trace_size > 0 and patch_idx == 0 else None,
        )
        for patch_idx in range(q_current_patches)
    ]
    workers = [norm_worker, *projection_workers]

    weight_taps = [
        [
            [
                TensorAccessPattern(
                    (weight_stream_bytes,),
                    (
                        (group_pos * q_current_patches + patch_idx) * k_chunks
                        + k_idx
                    )
                    * patch_bytes_per_k_chunk,
                    [1, 1, 1, patch_bytes_per_k_chunk],
                    [0, 0, 0, 1],
                )
                for k_idx in range(k_chunks)
            ]
            for patch_idx in range(q_current_patches)
        ]
        for group_pos in range(num_kv_groups)
    ]
    out_taps = [
        [
            TensorAccessPattern(
                (q_current_elements,),
                group_pos * q_current_elements_per_group + patch_idx * PATCH_ROWS,
                [1, 1, 1, PATCH_ROWS],
                [0, 0, 0, 1],
            )
            for patch_idx in range(q_current_patches)
        ]
        for group_pos in range(num_kv_groups)
    ]

    sequence_types = [hidden_ty, norm_weight_ty, weight_stream_ty, q_current_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        hidden, norm_weight, weight_stream, q_current = runtime_args[:4]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[projection_workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        for group_pos in range(num_kv_groups):
            tg = rt.task_group()
            if group_pos == 0:
                rt.fill(hidden_fifo.prod(), hidden, task_group=tg)
                rt.fill(norm_weight_fifo.prod(), norm_weight, task_group=tg)
            for patch_idx in range(q_current_patches):
                for k_idx in range(k_chunks):
                    rt.fill(
                        weight_fifos[patch_idx].prod(),
                        weight_stream,
                        weight_taps[group_pos][patch_idx][k_idx],
                        task_group=tg,
                    )
            for patch_idx in range(q_current_patches):
                rt.drain(
                    out_fifos[patch_idx].cons(),
                    q_current,
                    out_taps[group_pos][patch_idx],
                    wait=True,
                    task_group=tg,
                )
            rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
