# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer

from models.fast_qwen3.phase_tiles import (
    CHUNK_ROWS,
    PATCH_CHUNKS,
    PATCH_ROWS,
    projection_mem_tile,
    projection_shim_tile,
    projection_tile,
    residual_output_shim_tile,
    residual_tile,
)


def q4nx_fused_linear_residual_projection(
    dev,
    in_features,
    output_patches,
    trace_size=0,
    trace_ddr_id=7,
    func_prefix="",
    kernel_object="q4nx_fused_linear_residual_projection.o",
    verbose=False,
):
    if in_features % 256 != 0:
        raise ValueError("in_features must be divisible by 256")
    if not 1 <= output_patches <= 8:
        raise ValueError("output_patches must be in [1, 8]")
    if verbose:
        print(
            "Q4NX fused linear residual projection: "
            f"K={in_features}, output_patches={output_patches}"
        )

    chunk_bytes = 5120
    k_chunks = in_features // 256
    if k_chunks > 8:
        raise ValueError("linear residual projection currently supports at most 2048 K")
    weight_full_chunk_bytes = k_chunks * chunk_bytes
    weight_patch_bytes = PATCH_CHUNKS * weight_full_chunk_bytes
    weight_stream_bytes = output_patches * PATCH_CHUNKS * weight_full_chunk_bytes
    output_elements = output_patches * PATCH_ROWS

    input_ty = np.ndarray[(in_features,), np.dtype[bfloat16]]
    residual_ty = np.ndarray[(output_elements,), np.dtype[bfloat16]]
    weight_stream_ty = np.ndarray[(weight_stream_bytes,), np.dtype[np.uint8]]
    weight_patch_ty = np.ndarray[(weight_patch_bytes,), np.dtype[np.uint8]]
    weight_full_chunk_ty = np.ndarray[(weight_full_chunk_bytes,), np.dtype[np.uint8]]
    chunk_out_ty = np.ndarray[(CHUNK_ROWS,), np.dtype[bfloat16]]
    patch_out_ty = np.ndarray[(PATCH_ROWS,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(output_elements,), np.dtype[bfloat16]]

    input_fifo = ObjectFifo(input_ty, name="q4nx_linear_residual_input", depth=1)
    residual_fifo = ObjectFifo(
        residual_ty,
        name="q4nx_linear_residual_block",
        depth=1,
    )
    weight_patch_fifos = [
        ObjectFifo(
            weight_patch_ty,
            name=f"q4nx_linear_residual_weight_patch_{patch_idx}",
            depth=1,
        )
        for patch_idx in range(output_patches)
    ]
    weight_fifos = [
        weight_patch_fifos[patch_idx]
        .cons()
        .split(
            [chunk_idx * weight_full_chunk_bytes for chunk_idx in range(PATCH_CHUNKS)],
            obj_types=[weight_full_chunk_ty] * PATCH_CHUNKS,
            names=[
                f"q4nx_linear_residual_weight_{patch_idx}_{chunk_idx}"
                for chunk_idx in range(PATCH_CHUNKS)
            ],
            depths=[1] * PATCH_CHUNKS,
            placement=projection_mem_tile(patch_idx),
        )
        for patch_idx in range(output_patches)
    ]
    projection_patch_fifos = [
        ObjectFifo(
            patch_out_ty,
            name=f"q4nx_linear_residual_projected_patch_{patch_idx}",
            depth=1,
        )
        for patch_idx in range(output_patches)
    ]
    projection_chunk_fifos = [
        projection_patch_fifos[patch_idx]
        .prod()
        .join(
            [chunk_idx * CHUNK_ROWS for chunk_idx in range(PATCH_CHUNKS)],
            obj_types=[chunk_out_ty] * PATCH_CHUNKS,
            names=[
                f"q4nx_linear_residual_projected_{patch_idx}_{chunk_idx}"
                for chunk_idx in range(PATCH_CHUNKS)
            ],
            depths=[1] * PATCH_CHUNKS,
            placement=projection_mem_tile(patch_idx),
        )
        for patch_idx in range(output_patches)
    ]
    out_patch_fifos = [
        ObjectFifo(
            patch_out_ty,
            name=f"q4nx_linear_residual_out_patch_{patch_idx}",
            depth=1,
        )
        for patch_idx in range(output_patches)
    ]

    projection_kernel = Kernel(
        f"{func_prefix}q4nx_fused_projection_chunk_full",
        f"{func_prefix}{kernel_object}",
        [
            input_ty,
            weight_full_chunk_ty,
            chunk_out_ty,
        ],
    )
    residual_kernel = Kernel(
        f"{func_prefix}q4nx_residual_add_patch",
        f"{func_prefix}{kernel_object}",
        [
            patch_out_ty,
            residual_ty,
            np.int32,
            patch_out_ty,
        ],
    )

    def projection_body(input_fifo, weight_fifo, out_fifo, projection_kernel):
        x = input_fifo.acquire(1)
        weight = weight_fifo.acquire(1)
        out = out_fifo.acquire(1)
        projection_kernel(x, weight, out)
        out_fifo.release(1)
        weight_fifo.release(1)
        input_fifo.release(1)

    def residual_body(projected_fifo, residual_fifo, out_fifo, patch_idx, residual_kernel):
        projected = projected_fifo.acquire(1)
        residual = residual_fifo.acquire(1)
        out = out_fifo.acquire(1)
        residual_kernel(projected, residual, patch_idx * PATCH_ROWS, out)
        out_fifo.release(1)
        residual_fifo.release(1)
        projected_fifo.release(1)

    projection_workers = [
        Worker(
            projection_body,
            [
                input_fifo.cons(),
                weight_fifos[patch_idx][chunk_idx].cons(),
                projection_chunk_fifos[patch_idx][chunk_idx].prod(),
                projection_kernel,
            ],
            placement=projection_tile(patch_idx, chunk_idx),
            trace=(
                1
                if trace_size > 0 and patch_idx == 0 and chunk_idx == 0
                else None
            ),
        )
        for patch_idx in range(output_patches)
        for chunk_idx in range(PATCH_CHUNKS)
    ]
    residual_workers = [
        Worker(
            residual_body,
            [
                projection_patch_fifos[patch_idx].cons(),
                residual_fifo.cons(),
                out_patch_fifos[patch_idx].prod(),
                patch_idx,
                residual_kernel,
            ],
            placement=residual_tile(patch_idx),
        )
        for patch_idx in range(output_patches)
    ]
    workers = [*projection_workers, *residual_workers]

    weight_taps = [
        TensorAccessPattern(
            (weight_stream_bytes,),
            patch_idx * weight_patch_bytes,
            [1, 1, 1, weight_patch_bytes],
            [0, 0, 0, 1],
        )
        for patch_idx in range(output_patches)
    ]
    out_taps = [
        TensorAccessPattern(
            (output_elements,),
            patch_idx * PATCH_ROWS,
            [1, 1, 1, PATCH_ROWS],
            [0, 0, 0, 1],
        )
        for patch_idx in range(output_patches)
    ]

    sequence_types = [input_ty, residual_ty, weight_stream_ty, output_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        input_l3, residual_l3, weight_l3, output_l3 = runtime_args[:4]
        if trace_size > 0:
            rt.enable_trace(
                trace_size,
                workers=[projection_workers[0]],
                ddr_id=trace_ddr_id,
            )
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(
            input_fifo.prod(),
            input_l3,
            placement=Tile(0, 0),
            task_group=tg,
        )
        rt.fill(
            residual_fifo.prod(),
            residual_l3,
            placement=Tile(1, 0),
            task_group=tg,
        )
        for patch_idx in range(output_patches):
            rt.fill(
                weight_patch_fifos[patch_idx].prod(),
                weight_l3,
                weight_taps[patch_idx],
                placement=projection_shim_tile(patch_idx),
                task_group=tg,
            )
        for patch_idx in range(output_patches):
            rt.drain(
                out_patch_fifos[patch_idx].cons(),
                output_l3,
                out_taps[patch_idx],
                placement=residual_output_shim_tile(patch_idx),
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
