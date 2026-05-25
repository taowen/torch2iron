#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and run grouped attention over the FastFlowLM-style KV planes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from models.fast_qwen3.kv_plane_reference import (
    kv_plane_total_elements,
    plane_attention_current_reference,
)
from models.fast_qwen3.operators.qwen_plane_attention_current import (
    QwenPlaneAttentionCurrent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 plane-layout current-aware attention"
    )
    parser.add_argument("--build-dir", default="build/fast_qwen3_plane_attention")
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=64)
    parser.add_argument("--current-slot", type=int, default=63)
    parser.add_argument("--q-heads-per-group", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--plane-fifo-depth", type=int, default=2)
    parser.add_argument("--abs-tol", type=float, default=0.08)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be non-negative")
    if args.timed_iters <= 0:
        raise ValueError("timed-iters must be positive")

    torch.manual_seed(407895018)
    num_kv_groups = 8
    q_current_elements_per_group = (args.q_heads_per_group + 2) * args.head_dim
    q_current = torch.randn(
        num_kv_groups,
        q_current_elements_per_group,
        dtype=torch.bfloat16,
    )
    kv_plane = torch.randn(
        kv_plane_total_elements(args.packet_seq_len, args.head_dim),
        dtype=torch.bfloat16,
    )
    expected = plane_attention_current_reference(
        q_current,
        kv_plane,
        args.current_slot,
        args.attend_seq_len,
        args.packet_seq_len,
        args.q_heads_per_group,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    op = QwenPlaneAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.current_slot,
        q_heads_per_group=args.q_heads_per_group,
        head_dim=args.head_dim,
        tile_size=args.tile_size,
        plane_fifo_depth=args.plane_fifo_depth,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_plane_attention",
        runlist=[(op, "q_current", "kv_plane", "context")],
        input_args=["q_current", "kv_plane"],
        output_args=["context"],
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("q_current").torch_view().view_as(q_current)[:] = q_current
    fused.get_buffer("kv_plane").torch_view().view_as(kv_plane)[:] = kv_plane
    fused.mark_buffer_dirty("input")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual = fused.get_buffer("context").torch_view().view_as(expected)
    abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            f"plane attention max_abs_error={max_abs_error} exceeds {args.abs_tol}"
        )

    print(
        json.dumps(
            {
                "attend_seq_len": args.attend_seq_len,
                "build_dir": str(Path(args.build_dir).resolve()),
                "context_shape": list(actual.shape),
                "current_slot": args.current_slot,
                "head_dim": args.head_dim,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "packet_seq_len": args.packet_seq_len,
                "plane_fifo_depth": args.plane_fifo_depth,
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "q_heads_per_group": args.q_heads_per_group,
                "tile_size": args.tile_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
