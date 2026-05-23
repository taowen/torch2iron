# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_rms_norm(
    dev,
    num_elements,
    num_columns,
    num_channels,
    tile_size,
    trace_size,
    trace_ddr_id=4,
    func_prefix="",
    kernel_object="rms_norm.o",
):
    per_tile_elements = 8192 if tile_size > 8192 else tile_size
    total_cores = num_columns * num_channels
    per_core_elements = num_elements // total_cores
    if num_elements % total_cores != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {total_cores}."
        )
    N_div_n = per_core_elements // per_tile_elements
    chunk = num_elements // num_columns // num_channels  # For offset calculation
    dtype = bfloat16

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    fifodepth = 1 if tile_size > 4096 else 2

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # AIE Core Function declaration
    rms_norm_kernel = Kernel(
        f"{func_prefix}rms_norm_bf16_vector",
        f"{func_prefix}{kernel_object}",
        [tile_ty, tile_ty, np.int32],
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_out, rms_norm_kernel):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            rms_norm_kernel(elem_in1, elem_out, per_tile_elements)
            of_in1.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = []
    for i in range(num_columns):
        for j in range(num_channels):
            idx = i * num_channels + j
            my_workers.append(
                Worker(
                    core_body,
                    [
                        of_in1s[idx].cons(),
                        of_outs[idx].prod(),
                        rms_norm_kernel,
                    ],
                    trace=1 if trace_size > 0 and idx == 0 else None,
                )
            )

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    sequence_types = [tensor_ty, tensor_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        A, C = runtime_args[:2]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[my_workers[0]], ddr_id=trace_ddr_id)
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_in1s[i * num_channels + j].prod(),
                    A,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    C,
                    taps[i * num_channels + j],
                    wait=True,  # wait for the transfer to complete and data to be available
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())
