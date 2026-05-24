# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer


def my_w4a16_gemm(
    dev,
    cols,
    rows,
    M,
    K,
    N,
    tile_m,
    tile_k,
    tile_n,
    group_size,
    kernel_object="w4a16_gemm.o",
    func_prefix="",
    verbose=False,
    trace_size=0,
    trace_ddr_id=4,
):
    if verbose:
        print(
            "W4A16 GEMM: "
            f"M={M}, K={K}, N={N}, tile_m={tile_m}, tile_k={tile_k}, "
            f"tile_n={tile_n}, rows={rows}, cols={cols}"
        )

    assert tile_k == group_size
    assert M % (tile_m * rows) == 0
    assert K % tile_k == 0
    assert N % (tile_n * cols) == 0
    assert tile_k % 2 == 0
    mmul_r = 4
    mmul_s = 8
    mmul_t = 8
    assert tile_m == mmul_r or tile_m % (2 * mmul_r) == 0
    assert tile_k % mmul_s == 0
    assert tile_n % (2 * mmul_t) == 0

    k_tiles = K // tile_k
    m_tile_groups = M // (tile_m * rows)
    n_tiles = N // tile_n
    n_tile_groups = n_tiles // cols
    bf16_dtype = np.dtype[bfloat16]

    L1_A_ty = np.ndarray[(tile_m, tile_k), bf16_dtype]
    L1_QP_ty = np.ndarray[
        (tile_k // mmul_s, tile_n // mmul_t, mmul_s, mmul_t),
        bf16_dtype,
    ]
    L1_C_ty = np.ndarray[(tile_m, tile_n), bf16_dtype]
    L2_C_ty = np.ndarray[(rows * tile_m, tile_n), bf16_dtype]

    L3_A_ty = np.ndarray[(M, K), bf16_dtype]
    L3_QP_ty = np.ndarray[
        (cols, n_tile_groups, k_tiles, tile_k // mmul_s, tile_n // mmul_t, mmul_s, mmul_t),
        bf16_dtype,
    ]
    L3_C_ty = np.ndarray[(M, N), bf16_dtype]

    accum_kernel = Kernel(
        f"{func_prefix}w4a16_gemm_accum_bf16",
        f"{func_prefix}{kernel_object}",
        [np.int32, L1_A_ty, L1_QP_ty, L1_C_ty],
    )

    A_L3L2 = [
        ObjectFifo(L1_A_ty, name=f"A_L3L2_{row}", depth=2) for row in range(rows)
    ]
    A_L2L1 = [
        A_L3L2[row].cons().forward(
            obj_type=L1_A_ty,
            name=f"A_L2L1_{row}",
            depth=2,
            dims_to_stream=[
                (tile_m // mmul_r, mmul_r * tile_k),
                (tile_k // mmul_s, mmul_s),
                (mmul_r, tile_k),
                (mmul_s, 1),
            ],
            placement=Tile(row * 2 if cols == 8 else row, 1),
        )
        for row in range(rows)
    ]
    QP_L3L2 = [
        ObjectFifo(L1_QP_ty, name=f"QP_L3L2_{col}", depth=2)
        for col in range(cols)
    ]
    QP_L2L1 = [
        QP_L3L2[col].cons().forward(
            obj_type=L1_QP_ty,
            name=f"QP_L2L1_{col}",
            depth=2,
            placement=Tile(col, 1),
        )
        for col in range(cols)
    ]
    C_L2L3 = [
        ObjectFifo(
            L2_C_ty,
            name=f"C_L2L3_{col}",
            depth=2,
            dims_to_stream=[
                (tile_m // mmul_r, mmul_r * tile_n),
                (mmul_r, mmul_t),
                (tile_n // mmul_t, mmul_r * mmul_t),
                (mmul_t, 1),
            ],
        )
        for col in range(cols)
    ]
    C_L1L2 = [
        C_L2L3[col]
        .prod()
        .join(
            offsets=[row * tile_m * tile_n for row in range(rows)],
            obj_types=[L1_C_ty] * rows,
            names=[f"C_L1L2_{row}_{col}" for row in range(rows)],
            depths=[2] * rows,
            placement=Tile(col, 1),
        )
        for col in range(cols)
    ]

    def core_body(a_fifo, qp_fifo, c_fifo, accum_kernel):
        for _m_group in range(m_tile_groups):
            for _n_group in range_(n_tile_groups):
                c = c_fifo.acquire(1)
                for _k_tile in range(k_tiles):
                    a = a_fifo.acquire(1)
                    qp = qp_fifo.acquire(1)
                    accum_kernel(1 if _k_tile == 0 else 0, a, qp, c)
                    a_fifo.release(1)
                    qp_fifo.release(1)
                c_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_L2L1[row].cons(),
                QP_L2L1[col].cons(),
                C_L1L2[col][row].prod(),
                accum_kernel,
            ],
            placement=Tile(col, row + 2),
            trace=1 if trace_size > 0 and row == 0 and col == 0 else None,
        )
        for row in range(rows)
        for col in range(cols)
    ]

    sequence_types = [L3_A_ty, L3_QP_ty, L3_C_ty]
    if trace_size > 0:
        trace_ty = np.ndarray[(trace_size,), np.dtype[np.uint8]]
        sequence_types.extend([trace_ty] * max(1, trace_ddr_id - len(sequence_types) + 1))

    rt = Runtime()
    with rt.sequence(*sequence_types) as runtime_args:
        A, QP, C = runtime_args[:3]
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[workers[0]], ddr_id=trace_ddr_id)
        rt.start(*workers)
        for m_group in range(m_tile_groups):
            for n_group in range(n_tile_groups):
                tg = rt.task_group()
                for row in range(rows):
                    a_tap = TensorAccessPattern(
                        (M, K),
                        (m_group * rows + row) * tile_m * K,
                        [k_tiles, tile_m, tile_k],
                        [tile_k, K, 1],
                    )
                    rt.fill(A_L3L2[row].prod(), A, a_tap, task_group=tg)
                for col in range(cols):
                    qp_tap = TensorAccessPattern(
                        (cols * n_tile_groups * k_tiles * tile_n * tile_k,),
                        (
                            (col * n_tile_groups + n_group)
                            * k_tiles
                            * tile_n
                            * tile_k
                        ),
                        [1, 1, 1, k_tiles * tile_n * tile_k],
                        [0, 0, 0, 1],
                    )
                    rt.fill(QP_L3L2[col].prod(), QP, qp_tap, task_group=tg)
                for col in range(cols):
                    n_tile = n_group * cols + col
                    c_tap = TensorAccessPattern(
                        (M, N),
                        m_group * rows * tile_m * N + n_tile * tile_n,
                        [1, rows * tile_m, tile_n],
                        [0, N, 1],
                    )
                    rt.drain(
                        C_L2L3[col].cons(),
                        C,
                        c_tap,
                        wait=True,
                        task_group=tg,
                    )
                rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
