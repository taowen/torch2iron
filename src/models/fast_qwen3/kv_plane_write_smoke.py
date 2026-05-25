#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and run the FastFlowLM-style current K/V plane writer."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from models.fast_qwen3.kv_plane_reference import write_q_current_to_kv_plane
from models.fast_qwen3.operators.qwen_current_kv_plane_write import (
    QwenCurrentKVPlaneWrite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fast_qwen3 current K/V plane writer"
    )
    parser.add_argument("--build-dir", default="build/fast_qwen3_kv_plane_write")
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--current-slot", type=int, default=73)
    parser.add_argument("--q-heads-per-group", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be non-negative")
    if args.timed_iters <= 0:
        raise ValueError("timed-iters must be positive")

    torch.manual_seed(270445326)
    num_kv_groups = 8
    q_current_elements_per_group = (args.q_heads_per_group + 2) * args.head_dim
    q_current = torch.randn(
        num_kv_groups,
        q_current_elements_per_group,
        dtype=torch.bfloat16,
    )
    expected = write_q_current_to_kv_plane(
        q_current,
        args.current_slot,
        args.packet_seq_len,
        args.q_heads_per_group,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    op = QwenCurrentKVPlaneWrite(
        packet_seq_len=args.packet_seq_len,
        current_slot=args.current_slot,
        q_heads_per_group=args.q_heads_per_group,
        head_dim=args.head_dim,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_kv_plane_write",
        runlist=[(op, "q_current", "kv_plane")],
        input_args=["q_current"],
        output_args=["kv_plane"],
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("q_current").torch_view().view_as(q_current)[:] = q_current
    fused.mark_buffer_dirty("input")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual = fused.get_buffer("kv_plane").torch_view().view_as(expected)
    abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            f"kv plane write max_abs_error={max_abs_error} exceeds {args.abs_tol}"
        )

    print(
        json.dumps(
            {
                "build_dir": str(Path(args.build_dir).resolve()),
                "current_slot": args.current_slot,
                "head_dim": args.head_dim,
                "kv_plane_elements": int(expected.numel()),
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "packet_seq_len": args.packet_seq_len,
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "q_current_shape": list(q_current.shape),
                "q_heads_per_group": args.q_heads_per_group,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
