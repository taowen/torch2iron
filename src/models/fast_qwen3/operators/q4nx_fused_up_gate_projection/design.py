# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer


def q4nx_fused_up_gate_projection(
    dev,
    in_features,
    out_rows,
    output_patches,
    trace_size=0,
    trace_ddr_id=7,
    func_prefix="",
    kernel_object="q4nx_fused_up_gate_projection.o",
    verbose=False,
):
    if in_features % 256 != 0:
        raise ValueError("Q4NX fused up/gate patch requires in_features divisible by 256")
    if out_rows != 64:
        raise ValueError("Q4NX fused up/gate patch currently requires out_rows=64")
    if output_patches < 1 or output_patches > 8:
        raise ValueError("Q4NX fused up/gate patch currently requires 1..8 output patches")
    if verbose:
        print(
            "Q4NX fused up/gate patch: "
            f"K={in_features}, out_rows={out_rows}, output_patches={output_patches}"
        )

    chunk_bytes = 5120
    k_chunk_patch_bytes = 2 * chunk_bytes
    up_gate_k_chunk_bytes = 2 * k_chunk_patch_bytes
    weight_k_chunk_bytes = up_gate_k_chunk_bytes
    k_chunks = in_features // 256
    up_gate_patch_stream_bytes = output_patches * k_chunks * weight_k_chunk_bytes
    hidden_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    norm_weight_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    normed_hidden_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    up_gate_patch_stream_ty = np.ndarray[
        (up_gate_patch_stream_bytes,),
        np.dtype[np.uint8],
    ]
    weight_k_chunk_ty = np.ndarray[(weight_k_chunk_bytes,), np.dtype[np.uint8]]
    up_gate_patch_out_ty = np.ndarray[(2 * out_rows,), np.dtype[bfloat16]]
    up_gate_out_ty = np.ndarray[(output_patches * 2 * out_rows,), np.dtype[bfloat16]]

    hidden_fifo = ObjectFifo(hidden_ty, name="q4nx_up_gate_hidden", depth=1)
    norm_weight_fifo = ObjectFifo(
        norm_weight_ty,
        name="q4nx_up_gate_norm_weight",
        depth=1,
    )
    normed_hidden_l1l2 = ObjectFifo(
        normed_hidden_ty,
        name="q4nx_up_gate_normed_hidden_l1l2",
        depth=1,
    )
    normed_hidden_l2l1 = normed_hidden_l1l2.cons().forward(
        obj_type=normed_hidden_ty,
        name="q4nx_up_gate_normed_hidden_l2l1",
        depth=1,
        placement=Tile(0, 1),
    )
    up_gate_weight_fifos = [
        ObjectFifo(weight_k_chunk_ty, name=f"q4nx_up_gate_weight_{patch}", depth=1)
        for patch in range(output_patches)
    ]
    up_gate_out_fifos = [
        ObjectFifo(
            up_gate_patch_out_ty,
            name=f"q4nx_up_gate_out_{patch}",
            depth=1,
        )
        for patch in range(output_patches)
    ]

    norm_kernel = Kernel(
        f"{func_prefix}q4nx_up_gate_rms_norm_full",
        f"{func_prefix}{kernel_object}",
        [hidden_ty, norm_weight_ty, normed_hidden_ty],
    )
    up_gate_kernel = Kernel(
        f"{func_prefix}q4nx_fused_up_gate_projection_patch",
        f"{func_prefix}{kernel_object}",
        [
            np.int32,
            np.int32,
            normed_hidden_ty,
            weight_k_chunk_ty,
            up_gate_patch_out_ty,
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
        up_gate_weight_fifo,
        up_gate_out_fifo,
        up_gate_kernel,
    ):
        normed_hidden = normed_hidden_fifo.acquire(1)
        up_gate_out = up_gate_out_fifo.acquire(1)
        for k_idx in range(k_chunks):
            up_gate_weight = up_gate_weight_fifo.acquire(1)
            up_gate_kernel(
                1 if k_idx == 0 else 0,
                k_idx * 256,
                normed_hidden,
                up_gate_weight,
                up_gate_out,
            )
            up_gate_weight_fifo.release(1)
        up_gate_out_fifo.release(1)
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
    up_gate_workers = [
        Worker(
            core_body,
            [
                normed_hidden_l2l1.cons(),
                up_gate_weight_fifos[patch].cons(),
                up_gate_out_fifos[patch].prod(),
                up_gate_kernel,
            ],
            placement=Tile(patch, 2),
            trace=1 if trace_size > 0 and patch == 0 else None,
        )
        for patch in range(output_patches)
    ]
    workers = [norm_worker, *up_gate_workers]

    up_gate_weight_taps = [
        [
            TensorAccessPattern(
                (up_gate_patch_stream_bytes,),
                (patch * k_chunks + k_idx) * weight_k_chunk_bytes,
                [1, 1, 1, weight_k_chunk_bytes],
                [0, 0, 0, 1],
            )
            for k_idx in range(k_chunks)
        ]
        for patch in range(output_patches)
    ]
    up_gate_out_taps = [
        TensorAccessPattern(
            (output_patches * 2 * out_rows,),
            patch * 2 * out_rows,
            [1, 1, 1, 2 * out_rows],
            [0, 0, 0, 1],
        )
        for patch in range(output_patches)
    ]

    sequence_types = [
        hidden_ty,
        norm_weight_ty,
        up_gate_patch_stream_ty,
        up_gate_out_ty,
    ]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        hidden, norm_weight, up_gate_weight, up_gate_out = runtime_args[:4]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[up_gate_workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(hidden_fifo.prod(), hidden, task_group=tg)
        rt.fill(norm_weight_fifo.prod(), norm_weight, task_group=tg)
        for patch in range(output_patches):
            for k_idx in range(k_chunks):
                rt.fill(
                    up_gate_weight_fifos[patch].prod(),
                    up_gate_weight,
                    up_gate_weight_taps[patch][k_idx],
                    task_group=tg,
                )
        for patch in range(output_patches):
            rt.drain(
                up_gate_out_fifos[patch].cons(),
                up_gate_out,
                up_gate_out_taps[patch],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
