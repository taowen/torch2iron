# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def my_w4a16_matvec(
    dev,
    cols,
    M,
    K,
    group_size,
    m_input,
    m_output=None,
    num_batches=1,
    shared_qparam=False,
    kernel_object="w4a16_gemv.o",
    func_prefix="",
    verbose=False,
    trace_size=0,
    trace_ddr_id=4,
):
    if m_output is None:
        m_output = m_input
    if verbose:
        print(
            f"W4A16 GEMV: M={M}, K={K}, group={group_size}, cols={cols}, "
            f"batch={num_batches}, shared_qparam={shared_qparam}"
        )

    assert M % cols == 0
    assert K % 2 == 0
    assert K % group_size == 0
    assert m_output % m_input == 0 and m_output >= m_input
    assert m_output <= M // cols
    assert (M // cols) % m_output == 0
    assert m_input <= M // cols
    assert (M // cols) % m_input == 0

    num_groups = K // group_size
    qparam_row_bytes = K // 2 + num_groups * np.dtype(bfloat16).itemsize
    qweight_dtype = np.uint8
    bf16_dtype = np.dtype[bfloat16]

    L1_QP_ty = np.ndarray[(m_input, qparam_row_bytes), np.dtype[qweight_dtype]]
    L1_X_ty = np.ndarray[(K,), bf16_dtype]
    L1_Y_ty = np.ndarray[(m_output,), bf16_dtype]

    qparam_batches = 1 if shared_qparam else num_batches
    L3_QP_ty = np.ndarray[(qparam_batches * M * qparam_row_bytes,), np.dtype[qweight_dtype]]
    L3_X_ty = np.ndarray[(num_batches * K,), bf16_dtype]
    L3_Y_ty = np.ndarray[(num_batches * M,), bf16_dtype]

    kernel = Kernel(
        f"{func_prefix}w4a16_matvec_bf16",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, L1_QP_ty, L1_X_ty, L1_Y_ty],
    )

    QP_L3L1 = [ObjectFifo(L1_QP_ty, name=f"QP_L3L1_{i}", depth=2) for i in range(cols)]
    X_L3L1 = [ObjectFifo(L1_X_ty, name=f"X_L3L1_{i}", depth=1) for i in range(cols)]
    Y_L1L3 = [ObjectFifo(L1_Y_ty, name=f"Y_L1L3_{i}", depth=2) for i in range(cols)]

    def core_body(qp_fifo, x_fifo, y_fifo, kernel):
        x = x_fifo.acquire(1)
        for i_idx in range_(M // m_output // cols):
            y = y_fifo.acquire(1)
            for j_idx in range_(m_output // m_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * m_input
                qp = qp_fifo.acquire(1)
                kernel(m_input, output_row_offset, qp, x, y)
                qp_fifo.release(1)
            y_fifo.release(1)
        x_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                QP_L3L1[i].cons(),
                X_L3L1[i].cons(),
                Y_L1L3[i].prod(),
                kernel,
            ],
            trace=1 if trace_size > 0 and i == 0 else None,
        )
        for i in range(cols)
    ]

    QP_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_QP_ty.__args__[0],
                offset=(
                    col * (M // cols) * qparam_row_bytes
                    + (0 if shared_qparam else batch * M * qparam_row_bytes)
                ),
                sizes=[1, 1, 1, (M // cols) * qparam_row_bytes],
                strides=[0, 0, 0, 1],
            )
            for batch in range(num_batches)
        ]
        for col in range(cols)
    ]
    X_tap = TensorAccessPattern(
        tensor_dims=L3_X_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, num_batches * K],
        strides=[0, 0, 0, 1],
    )
    Y_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_Y_ty.__args__[0],
                offset=col * (M // cols) + batch * M,
                sizes=[1, 1, 1, M // cols],
                strides=[0, 0, 0, 1],
            )
            for batch in range(num_batches)
        ]
        for col in range(cols)
    ]

    sequence_types = [L3_QP_ty, L3_X_ty, L3_Y_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        QP, X, Y = runtime_args[:3]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        tg_x = rt.task_group()
        for col in range(cols):
            rt.fill(X_L3L1[col].prod(), X, X_tap, task_group=tg_x)
        for batch in range(num_batches):
            tg = rt.task_group()
            for col in range(cols):
                rt.fill(QP_L3L1[col].prod(), QP, QP_taps[col][batch], task_group=tg)
            for col in range(cols):
                rt.drain(Y_L1L3[col].cons(), Y, Y_taps[col][batch], task_group=tg, wait=True)
            rt.finish_task_group(tg)
        rt.finish_task_group(tg_x)

    return Program(dev, rt).resolve_program(SequentialPlacer())
