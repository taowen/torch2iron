#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one-read two-stage bounded static-KV attention and compare context output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

from models.fast_qwen3.kv_plane_reference import (
    kv_plane_total_elements,
)
from models.fast_qwen3.operators.qwen3_layer_fused.static_kv_reader import (
    HEAD_DIM,
    PLANES,
    Q_ELEMENTS_PER_GROUP,
    Q_HEADS_PER_GROUP,
    history_tiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-read two-stage bounded static-KV attention"
    )
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--insts", type=Path, required=True)
    parser.add_argument("--kernel", default="MLIR_AIE")
    parser.add_argument("--attend-seq-len", type=int, default=128)
    parser.add_argument("--packet-seq-len", type=int, default=4096)
    parser.add_argument("--abs-tol", type=float, default=0.08)
    return parser.parse_args()


def _deterministic_kv_plane(packet_seq_len: int) -> torch.Tensor:
    elements = kv_plane_total_elements(packet_seq_len, HEAD_DIM)
    values = (torch.arange(elements, dtype=torch.float32) % 257 - 128.0) / 512.0
    return values.to(torch.bfloat16)


def _zero_query_reference(
    kv_plane: torch.Tensor,
    attend_seq_len: int,
) -> torch.Tensor:
    output = torch.empty(
        4,
        Q_HEADS_PER_GROUP,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    tile_count = history_tiles(attend_seq_len)
    value_plane_offset = PLANES[1].base_bytes // 2
    for group in range(4):
        half = group % 2
        values = []
        for tile in range(tile_count):
            valid_rows = min(16, attend_seq_len - tile * 16)
            offset = value_plane_offset + tile * 4096 + half * 2048
            chunk = kv_plane[offset : offset + 2048].view(16, HEAD_DIM)
            values.append(chunk[:valid_rows].to(torch.float32))
        mean_value = torch.cat(values, dim=0).mean(dim=0).to(torch.bfloat16)
        for q_head in range(Q_HEADS_PER_GROUP):
            output[group, q_head] = mean_value
    return output


def _bf16_from_u16(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data.copy()).view(torch.bfloat16)


def main() -> None:
    args = parse_args()
    if args.attend_seq_len <= 0 or args.attend_seq_len > args.packet_seq_len:
        raise ValueError("attend-seq-len must be in (0, packet-seq-len]")

    kv_plane = _deterministic_kv_plane(args.packet_seq_len)
    expected = _zero_query_reference(
        kv_plane,
        args.attend_seq_len,
    )

    kv_cache = XRTTensor((kv_plane.numel(),), dtype=np.uint16)
    kv_cache.data[:] = kv_plane.contiguous().view(torch.uint16).numpy()
    kv_cache.to("npu")

    context = XRTTensor((8 * Q_ELEMENTS_PER_GROUP,), dtype=np.uint16)
    context.data.fill(0x7FC0)
    context.to("npu")

    kernel = NPUKernel(
        xclbin_path=args.xclbin,
        insts_path=args.insts,
        kernel_name=args.kernel,
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    result = aie_utils.DefaultNPURuntime.run(handle, [kv_cache, context])
    context.to("cpu")

    actual = _bf16_from_u16(context.data).view(8, Q_HEADS_PER_GROUP, HEAD_DIM)[:4]
    abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    if not result.is_success():
        raise RuntimeError(f"NPU run failed: {result.ret}")
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            "one-read two-stage bounded static-KV attention "
            f"max_abs_error={max_abs_error} exceeds {args.abs_tol}"
        )

    print(
        json.dumps(
            {
                "attend_seq_len": args.attend_seq_len,
                "context_shape": list(actual.shape),
                "kernel": args.kernel,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "npu_time_ns": result.npu_time,
                "packet_seq_len": args.packet_seq_len,
                "ret": str(result.ret),
                "success": result.is_success(),
                "xclbin": str(args.xclbin.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
