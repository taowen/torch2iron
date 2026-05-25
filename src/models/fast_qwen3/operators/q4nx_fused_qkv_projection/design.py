# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer


def q4nx_fused_qkv_projection(
    dev,
    in_features,
    out_rows,
    output_patches,
    trace_size=0,
    trace_ddr_id=7,
    func_prefix="",
    kernel_object="q4nx_fused_qkv_projection.o",
    verbose=False,
):
    if in_features % 256 != 0:
        raise ValueError("Q4NX fused QKV patch requires in_features divisible by 256")
    if out_rows != 64:
        raise ValueError("Q4NX fused QKV patch currently requires out_rows=64")
    if output_patches < 1 or output_patches > 8:
        raise ValueError("Q4NX fused QKV patch currently requires 1..8 output patches")
    if verbose:
        print(
            "Q4NX fused QKV patch: "
            f"K={in_features}, out_rows={out_rows}, output_patches={output_patches}"
        )

    chunk_bytes = 5120
    k_chunk_patch_bytes = 2 * chunk_bytes
    qkv_k_chunk_bytes = 3 * k_chunk_patch_bytes
    weight_k_chunk_bytes = qkv_k_chunk_bytes
    k_chunks = in_features // 256
    qkv_patch_stream_bytes = output_patches * k_chunks * weight_k_chunk_bytes
    hidden_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    norm_weight_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    normed_hidden_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    qkv_patch_stream_ty = np.ndarray[(qkv_patch_stream_bytes,), np.dtype[np.uint8]]
    weight_k_chunk_ty = np.ndarray[(weight_k_chunk_bytes,), np.dtype[np.uint8]]
    qkv_patch_out_ty = np.ndarray[(3 * out_rows,), np.dtype[bfloat16]]
    qkv_out_ty = np.ndarray[(output_patches * 3 * out_rows,), np.dtype[bfloat16]]

    hidden_fifo = ObjectFifo(hidden_ty, name="q4nx_qkv_hidden", depth=1)
    norm_weight_fifo = ObjectFifo(
        norm_weight_ty,
        name="q4nx_qkv_norm_weight",
        depth=1,
    )
    normed_hidden_l1l2 = ObjectFifo(
        normed_hidden_ty,
        name="q4nx_qkv_normed_hidden_l1l2",
        depth=1,
    )
    normed_hidden_l2l1 = normed_hidden_l1l2.cons().forward(
        obj_type=normed_hidden_ty,
        name="q4nx_qkv_normed_hidden_l2l1",
        depth=1,
        placement=Tile(0, 1),
    )
    qkv_weight_fifos = [
        ObjectFifo(weight_k_chunk_ty, name=f"q4nx_qkv_weight_{patch_idx}", depth=1)
        for patch_idx in range(output_patches)
    ]
    qkv_out_fifos = [
        ObjectFifo(qkv_patch_out_ty, name=f"q4nx_qkv_out_{patch_idx}", depth=1)
        for patch_idx in range(output_patches)
    ]

    norm_kernel = Kernel(
        f"{func_prefix}q4nx_rms_norm_full",
        f"{func_prefix}{kernel_object}",
        [hidden_ty, norm_weight_ty, normed_hidden_ty],
    )
    qkv_kernel = Kernel(
        f"{func_prefix}q4nx_fused_qkv_projection_patch",
        f"{func_prefix}{kernel_object}",
        [
            np.int32,
            np.int32,
            normed_hidden_ty,
            weight_k_chunk_ty,
            qkv_patch_out_ty,
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

    def core_body(
        normed_hidden_fifo,
        qkv_weight_fifo,
        qkv_out_fifo,
        qkv_kernel,
    ):
        normed_hidden = normed_hidden_fifo.acquire(1)
        qkv_out = qkv_out_fifo.acquire(1)
        for k_idx in range(k_chunks):
            qkv_weight = qkv_weight_fifo.acquire(1)
            qkv_kernel(
                1 if k_idx == 0 else 0,
                k_idx * 256,
                normed_hidden,
                qkv_weight,
                qkv_out,
            )
            qkv_weight_fifo.release(1)
        qkv_out_fifo.release(1)
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
    qkv_workers = [
        Worker(
            core_body,
            [
                normed_hidden_l2l1.cons(),
                qkv_weight_fifos[patch_idx].cons(),
                qkv_out_fifos[patch_idx].prod(),
                qkv_kernel,
            ],
            placement=Tile(patch_idx, 2),
            trace=1 if trace_size > 0 and patch_idx == 0 else None,
        )
        for patch_idx in range(output_patches)
    ]
    workers = [norm_worker, *qkv_workers]

    qkv_weight_taps = [
        [
            TensorAccessPattern(
                (qkv_patch_stream_bytes,),
                (patch_idx * k_chunks + k_idx) * weight_k_chunk_bytes,
                [1, 1, 1, weight_k_chunk_bytes],
                [0, 0, 0, 1],
            )
            for k_idx in range(k_chunks)
        ]
        for patch_idx in range(output_patches)
    ]
    qkv_out_taps = [
        TensorAccessPattern(
            (output_patches * 3 * out_rows,),
            patch_idx * 3 * out_rows,
            [1, 1, 1, 3 * out_rows],
            [0, 0, 0, 1],
        )
        for patch_idx in range(output_patches)
    ]

    sequence_types = [hidden_ty, norm_weight_ty, qkv_patch_stream_ty, qkv_out_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        hidden, norm_weight, qkv_weight, qkv_out = runtime_args[:4]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[qkv_workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(hidden_fifo.prod(), hidden, task_group=tg)
        rt.fill(norm_weight_fifo.prod(), norm_weight, task_group=tg)
        for patch_idx in range(output_patches):
            for k_idx in range(k_chunks):
                rt.fill(
                    qkv_weight_fifos[patch_idx].prod(),
                    qkv_weight,
                    qkv_weight_taps[patch_idx][k_idx],
                    task_group=tg,
                )
        for patch_idx in range(output_patches):
            rt.drain(
                qkv_out_fifos[patch_idx].cons(),
                qkv_out,
                qkv_out_taps[patch_idx],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
