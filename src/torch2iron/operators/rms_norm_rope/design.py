# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer


def weighted_rms_norm_rope(
    dev,
    rows,
    cols,
    angle_rows,
    num_aie_columns,
    trace_size=0,
    trace_ddr_id=4,
    func_prefix="",
    kernel_object="weighted_rms_norm_rope.o",
):
    if cols % 32 != 0:
        raise ValueError("cols must be divisible by 32")
    if rows % num_aie_columns != 0:
        raise ValueError("rows must be divisible by num_aie_columns")
    if angle_rows > rows or rows % angle_rows != 0:
        raise ValueError("angle_rows must divide rows")
    if angle_rows < num_aie_columns or angle_rows % num_aie_columns != 0:
        raise ValueError("angle_rows must be divisible by num_aie_columns")

    dtype = bfloat16
    tensor_rows_per_column = rows // num_aie_columns
    angle_rows_per_column = angle_rows // num_aie_columns
    tensor_rows_per_angle = rows // angle_rows

    tensor_ty = np.ndarray[(rows, cols), np.dtype[dtype]]
    weight_ty = np.ndarray[(cols,), np.dtype[dtype]]
    angle_ty = np.ndarray[(angle_rows, cols), np.dtype[dtype]]
    row_ty = np.ndarray[(1, cols), np.dtype[dtype]]

    input_fifos = [
        ObjectFifo(row_ty, name=f"rms_rope_in_{col}", depth=2)
        for col in range(num_aie_columns)
    ]
    norm_fifos = [
        ObjectFifo(row_ty, name=f"rms_rope_norm_{col}", depth=2)
        for col in range(num_aie_columns)
    ]
    angle_fifos = [
        ObjectFifo(row_ty, name=f"rms_rope_angle_{col}", depth=2)
        for col in range(num_aie_columns)
    ]
    output_fifos = [
        ObjectFifo(row_ty, name=f"rms_rope_out_{col}", depth=2)
        for col in range(num_aie_columns)
    ]
    weight_fifo = ObjectFifo(weight_ty, name="rms_rope_weight", depth=2)

    norm_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_row_bf16",
        f"{func_prefix}{kernel_object}",
        [row_ty, weight_ty, row_ty, np.int32],
    )
    rope_kernel = Kernel(
        f"{func_prefix}rope_row_bf16",
        f"{func_prefix}{kernel_object}",
        [row_ty, row_ty, row_ty, np.int32],
    )

    def norm_body(input_fifo, weight_fifo, norm_fifo, norm_kernel):
        weight = weight_fifo.acquire(1)
        for _ in range_(tensor_rows_per_column):
            input_row = input_fifo.acquire(1)
            norm_row = norm_fifo.acquire(1)
            norm_kernel(input_row, weight, norm_row, cols)
            input_fifo.release(1)
            norm_fifo.release(1)
        weight_fifo.release(1)

    def rope_body(norm_fifo, angle_fifo, output_fifo, rope_kernel):
        for _ in range_(angle_rows_per_column):
            angle = angle_fifo.acquire(1)
            for _ in range_(tensor_rows_per_angle):
                norm_row = norm_fifo.acquire(1)
                output_row = output_fifo.acquire(1)
                rope_kernel(norm_row, angle, output_row, cols)
                norm_fifo.release(1)
                output_fifo.release(1)
            angle_fifo.release(1)

    norm_workers = [
        Worker(
            norm_body,
            [
                input_fifos[col].cons(),
                weight_fifo.cons(),
                norm_fifos[col].prod(),
                norm_kernel,
            ],
            trace=1 if trace_size > 0 and col == 0 else None,
        )
        for col in range(num_aie_columns)
    ]
    rope_workers = [
        Worker(
            rope_body,
            [
                norm_fifos[col].cons(),
                angle_fifos[col].cons(),
                output_fifos[col].prod(),
                rope_kernel,
            ],
        )
        for col in range(num_aie_columns)
    ]
    workers = [*norm_workers, *rope_workers]

    tensor_taps = [
        TensorAccessPattern(
            (rows, cols),
            col * tensor_rows_per_column * cols,
            [1, 1, 1, tensor_rows_per_column * cols],
            [0, 0, 0, 1],
        )
        for col in range(num_aie_columns)
    ]
    angle_taps = [
        TensorAccessPattern(
            (angle_rows, cols),
            col * angle_rows_per_column * cols,
            [1, 1, 1, angle_rows_per_column * cols],
            [0, 0, 0, 1],
        )
        for col in range(num_aie_columns)
    ]

    sequence_types = [tensor_ty, weight_ty, angle_ty, tensor_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        input_tensor, weight, angles, output = runtime_args[:4]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        tg = rt.task_group()
        for col in range(num_aie_columns):
            rt.fill(
                input_fifos[col].prod(),
                input_tensor,
                tensor_taps[col],
                task_group=tg,
            )
            rt.fill(
                angle_fifos[col].prod(),
                angles,
                angle_taps[col],
                task_group=tg,
            )
        rt.fill(weight_fifo.prod(), weight, task_group=tg)
        for col in range(num_aie_columns):
            rt.drain(
                output_fifos[col].cons(),
                output,
                tensor_taps[col],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
