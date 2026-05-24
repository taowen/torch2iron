# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer


def residual_add_rms_norm(
    dev,
    num_elements,
    num_columns,
    tile_size,
    trace_size=0,
    trace_ddr_id=5,
    func_prefix="",
    kernel_object="residual_add_rms_norm.o",
):
    if num_elements % (num_columns * tile_size) != 0:
        raise ValueError("num_elements must be divisible by num_columns * tile_size")

    rows_per_column = num_elements // (num_columns * tile_size)
    chunk = num_elements // num_columns
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    weight_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    residual_fifos = [
        ObjectFifo(tile_ty, name=f"residual_in_{col}", depth=2)
        for col in range(num_columns)
    ]
    update_fifos = [
        ObjectFifo(tile_ty, name=f"update_in_{col}", depth=2)
        for col in range(num_columns)
    ]
    sum_fifos = [
        ObjectFifo(tile_ty, name=f"sum_out_{col}", depth=2)
        for col in range(num_columns)
    ]
    norm_input_fifos = [
        ObjectFifo(tile_ty, name=f"norm_in_{col}", depth=2)
        for col in range(num_columns)
    ]
    norm_fifos = [
        ObjectFifo(tile_ty, name=f"norm_out_{col}", depth=2)
        for col in range(num_columns)
    ]
    weight_fifo = ObjectFifo(weight_ty, name="norm_weight", depth=2)

    add_kernel = Kernel(
        f"{func_prefix}residual_add_bf16_vector",
        f"{func_prefix}{kernel_object}",
        [tile_ty, tile_ty, tile_ty, tile_ty, np.int32],
    )
    norm_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_bf16_vector",
        f"{func_prefix}{kernel_object}",
        [tile_ty, weight_ty, tile_ty, np.int32],
    )

    def add_body(
        residual_fifo,
        update_fifo,
        sum_fifo,
        norm_input_fifo,
        add_kernel,
    ):
        for _ in range_(rows_per_column):
            residual = residual_fifo.acquire(1)
            update = update_fifo.acquire(1)
            summed = sum_fifo.acquire(1)
            norm_input = norm_input_fifo.acquire(1)
            add_kernel(residual, update, summed, norm_input, tile_size)
            residual_fifo.release(1)
            update_fifo.release(1)
            sum_fifo.release(1)
            norm_input_fifo.release(1)

    def norm_body(norm_input_fifo, weight_fifo, norm_fifo, norm_kernel):
        weight = weight_fifo.acquire(1)
        for _ in range_(rows_per_column):
            norm_input = norm_input_fifo.acquire(1)
            norm = norm_fifo.acquire(1)
            norm_kernel(norm_input, weight, norm, tile_size)
            norm_input_fifo.release(1)
            norm_fifo.release(1)
        weight_fifo.release(1)

    add_workers = [
        Worker(
            add_body,
            [
                residual_fifos[col].cons(),
                update_fifos[col].cons(),
                sum_fifos[col].prod(),
                norm_input_fifos[col].prod(),
                add_kernel,
            ],
            trace=1 if trace_size > 0 and col == 0 else None,
        )
        for col in range(num_columns)
    ]
    norm_workers = [
        Worker(
            norm_body,
            [
                norm_input_fifos[col].cons(),
                weight_fifo.cons(),
                norm_fifos[col].prod(),
                norm_kernel,
            ],
        )
        for col in range(num_columns)
    ]
    workers = [*add_workers, *norm_workers]

    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * col,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for col in range(num_columns)
    ]

    sequence_types = [tensor_ty, tensor_ty, weight_ty, tensor_ty, tensor_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        residual, update, weight, summed, norm = runtime_args[:5]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        tg = rt.task_group()
        for col in range(num_columns):
            rt.fill(
                residual_fifos[col].prod(),
                residual,
                taps[col],
                task_group=tg,
            )
            rt.fill(
                update_fifos[col].prod(),
                update,
                taps[col],
                task_group=tg,
            )
        rt.fill(weight_fifo.prod(), weight, task_group=tg)
        for col in range(num_columns):
            rt.drain(
                sum_fifos[col].cons(),
                summed,
                taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                norm_fifos[col].cons(),
                norm,
                taps[col],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
