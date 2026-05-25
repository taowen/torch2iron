# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.placers import SequentialPlacer


PATCH_ROWS = 64


def _tap(buffer_elements, offset, length):
    return TensorAccessPattern(
        (buffer_elements,),
        offset,
        [1, 1, 1, length],
        [0, 0, 0, 1],
    )


def _copy_once(
    rt,
    fifo_in,
    fifo_out,
    src,
    dst,
    src_elements,
    dst_elements,
    src_offset,
    dst_offset,
):
    tg = rt.task_group()
    rt.fill(
        fifo_in.prod(),
        src,
        _tap(src_elements, src_offset, PATCH_ROWS),
        task_group=tg,
    )
    rt.drain(
        fifo_out.cons(),
        dst,
        _tap(dst_elements, dst_offset, PATCH_ROWS),
        wait=True,
        task_group=tg,
    )
    rt.finish_task_group(tg)


def qwen_qkv_to_q_current(
    dev,
    qkv_output_patches,
    num_kv_groups,
    q_heads_per_group,
    head_dim,
    verbose=False,
):
    if head_dim % PATCH_ROWS != 0:
        raise ValueError("head_dim must be divisible by 64")
    if verbose:
        print(
            "qwen_qkv_to_q_current: "
            f"qkv_output_patches={qkv_output_patches}, "
            f"num_kv_groups={num_kv_groups}, q_heads_per_group={q_heads_per_group}, "
            f"head_dim={head_dim}"
        )

    head_chunks = head_dim // PATCH_ROWS
    q_elements_per_group = q_heads_per_group * head_dim
    q_current_elements_per_group = q_elements_per_group + 2 * head_dim
    qkv_elements = qkv_output_patches * 3 * PATCH_ROWS
    q_current_elements = num_kv_groups * q_current_elements_per_group
    dtype = bfloat16

    qkv_ty = np.ndarray[(qkv_elements,), np.dtype[dtype]]
    q_current_ty = np.ndarray[(q_current_elements,), np.dtype[dtype]]
    patch_ty = np.ndarray[(PATCH_ROWS,), np.dtype[dtype]]
    patch_fifo = ObjectFifo(patch_ty, name="qwen_qkv_to_q_current_patch", depth=1)
    patch_out = patch_fifo.cons().forward(
        name="qwen_qkv_to_q_current_patch_out",
        depth=1,
    )

    rt = Runtime()
    with rt.sequence(qkv_ty, q_current_ty) as (qkv, q_current):
        for group in range(num_kv_groups):
            group_q_base = group * q_heads_per_group * head_chunks
            group_kv_base = group * head_chunks
            dst_group_base = group * q_current_elements_per_group
            for q_head in range(q_heads_per_group):
                for head_chunk in range(head_chunks):
                    src_patch = group_q_base + q_head * head_chunks + head_chunk
                    dst_offset = dst_group_base + q_head * head_dim + head_chunk * PATCH_ROWS
                    _copy_once(
                        rt,
                        patch_fifo,
                        patch_out,
                        qkv,
                        q_current,
                        qkv_elements,
                        q_current_elements,
                        src_patch * 3 * PATCH_ROWS,
                        dst_offset,
                    )
            for head_chunk in range(head_chunks):
                src_patch = group_kv_base + head_chunk
                dst_offset = (
                    dst_group_base + q_elements_per_group + head_chunk * PATCH_ROWS
                )
                _copy_once(
                    rt,
                    patch_fifo,
                    patch_out,
                    qkv,
                    q_current,
                    qkv_elements,
                    q_current_elements,
                    src_patch * 3 * PATCH_ROWS + PATCH_ROWS,
                    dst_offset,
                )
            for head_chunk in range(head_chunks):
                src_patch = group_kv_base + head_chunk
                dst_offset = (
                    dst_group_base
                    + q_elements_per_group
                    + head_dim
                    + head_chunk * PATCH_ROWS
                )
                _copy_once(
                    rt,
                    patch_fifo,
                    patch_out,
                    qkv,
                    q_current,
                    qkv_elements,
                    q_current_elements,
                    src_patch * 3 * PATCH_ROWS + 2 * PATCH_ROWS,
                    dst_offset,
                )

    return Program(dev, rt).resolve_program(SequentialPlacer())
