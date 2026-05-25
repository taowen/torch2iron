#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run current-K/V plane write directly into plane-layout attention."""

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
    update_q_current_in_kv_plane,
)
from models.fast_qwen3.operators.qwen_current_kv_plane_write import (
    QwenCurrentKVPlaneWrite,
)
from models.fast_qwen3.operators.qwen_plane_attention_current import (
    QwenPlaneAttentionCurrent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 current K/V write into next-token attention"
    )
    parser.add_argument(
        "--build-dir",
        default="build/fast_qwen3_kv_plane_write_attention",
    )
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--write-slot", type=int, default=0)
    parser.add_argument("--attention-slot", type=int, default=1)
    parser.add_argument("--attend-seq-len", type=int, default=2)
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
    if args.write_slot < 0 or args.write_slot >= args.packet_seq_len:
        raise ValueError("write-slot must be inside packet-seq-len")
    if args.attention_slot < 0 or args.attention_slot >= args.attend_seq_len:
        raise ValueError("attention-slot must be inside attend-seq-len")
    if args.write_slot >= args.attend_seq_len:
        raise ValueError("write-slot must be visible to the attention window")
    if args.write_slot == args.attention_slot:
        raise ValueError("write-slot must be a history slot for this smoke")

    torch.manual_seed(1834659607)
    num_kv_groups = 8
    q_current_elements_per_group = (args.q_heads_per_group + 2) * args.head_dim
    write_q_current = torch.randn(
        num_kv_groups,
        q_current_elements_per_group,
        dtype=torch.bfloat16,
    )
    attn_q_current = torch.randn(
        num_kv_groups,
        q_current_elements_per_group,
        dtype=torch.bfloat16,
    )

    kv_plane = torch.randn(
        kv_plane_total_elements(args.packet_seq_len, args.head_dim),
        dtype=torch.bfloat16,
    )
    written_plane = kv_plane.clone()
    update_q_current_in_kv_plane(
        write_q_current,
        written_plane,
        args.write_slot,
        args.packet_seq_len,
        args.q_heads_per_group,
    )
    expected_context = plane_attention_current_reference(
        attn_q_current,
        written_plane,
        args.attention_slot,
        args.attend_seq_len,
        args.packet_seq_len,
        args.q_heads_per_group,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    write_op = QwenCurrentKVPlaneWrite(
        packet_seq_len=args.packet_seq_len,
        current_slot=args.write_slot,
        q_heads_per_group=args.q_heads_per_group,
        head_dim=args.head_dim,
        context=context,
    )
    attention_op = QwenPlaneAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.attention_slot,
        q_heads_per_group=args.q_heads_per_group,
        head_dim=args.head_dim,
        tile_size=args.tile_size,
        plane_fifo_depth=args.plane_fifo_depth,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_kv_plane_write_attention",
        runlist=[
            (write_op, "write_q_current", "kv_plane"),
            (attention_op, "attn_q_current", "kv_plane", "context"),
        ],
        input_args=["write_q_current", "attn_q_current"],
        output_args=["context"],
        external_args={"kv_plane": ["kv_plane"]},
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("write_q_current").torch_view().view_as(write_q_current)[
        :
    ] = write_q_current
    fused.get_buffer("attn_q_current").torch_view().view_as(attn_q_current)[
        :
    ] = attn_q_current
    fused.get_buffer("kv_plane").torch_view().view_as(kv_plane)[:] = kv_plane
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("kv_plane")
    fused.kv_plane_buffer.to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual_context = (
        fused.get_buffer("context")
        .torch_view()
        .view(num_kv_groups, args.q_heads_per_group, args.head_dim)
    )
    fused.kv_plane_buffer.to("cpu")
    actual_plane = fused.get_buffer("kv_plane").torch_view().view_as(written_plane)
    abs_error = (
        actual_context.to(torch.float32) - expected_context.to(torch.float32)
    ).abs()
    max_abs_error = float(abs_error.max().item())
    plane_abs_error = (
        actual_plane.to(torch.float32) - written_plane.to(torch.float32)
    ).abs()
    max_plane_abs_error = float(plane_abs_error.max().item())
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            f"write -> plane attention max_abs_error={max_abs_error}"
        )
    if max_plane_abs_error != 0.0:
        raise AssertionError(f"kv_plane in-place max_abs_error={max_plane_abs_error}")

    print(
        json.dumps(
            {
                "attention_slot": args.attention_slot,
                "attend_seq_len": args.attend_seq_len,
                "build_dir": str(Path(args.build_dir).resolve()),
                "context_shape": list(actual_context.shape),
                "head_dim": args.head_dim,
                "max_abs_error": max_abs_error,
                "mean_abs_error": float(abs_error.mean().item()),
                "packet_seq_len": args.packet_seq_len,
                "plane_max_abs_error": max_plane_abs_error,
                "plane_fifo_depth": args.plane_fifo_depth,
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "q_current_shape": list(attn_q_current.shape),
                "q_heads_per_group": args.q_heads_per_group,
                "tile_size": args.tile_size,
                "write_slot": args.write_slot,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
