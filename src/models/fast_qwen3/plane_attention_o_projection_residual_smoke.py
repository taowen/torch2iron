#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run plane-layout attention into packed-Q4 O projection plus residual."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from models.fast_qwen3.attention_o_projection_residual_smoke import (
    _linear_prefix,
    _linear_weight_stream,
)
from models.fast_qwen3.fast_packed_format import (
    FastQwen3Store,
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.kv_plane_reference import (
    kv_plane_total_elements,
    plane_attention_current_reference,
)
from models.fast_qwen3.operators.q4nx_fused_linear_residual_projection import (
    Q4NXFusedLinearResidualProjection,
)
from models.fast_qwen3.operators.qwen_plane_attention_current import (
    QwenPlaneAttentionCurrent,
)
from models.fast_qwen3.q_current_plane_write_attention_smoke import _artifact_dir
from models.fast_qwen3.q4nx_layout import (
    Q4NX_PATCH_OUT_ROWS,
    q4nx_output_patch_reference,
)
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 plane attention through O projection residual"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--build-dir",
        default="build/fast_qwen3_plane_attention_o_projection_residual",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--o-block", type=int, default=0)
    parser.add_argument("--output-patches", type=int, default=8)
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=64)
    parser.add_argument("--current-slot", type=int, default=63)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--plane-fifo-depth", type=int, default=2)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.12)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be non-negative")
    if args.timed_iters <= 0:
        raise ValueError("timed-iters must be positive")
    if not 1 <= args.output_patches <= 8:
        raise ValueError("output-patches must be in [1, 8]")

    model_dir = find_model_dir(args.model_dir)
    artifact_dir = _artifact_dir(model_dir, args.artifact_dir)
    if args.repack or not (artifact_dir / "manifest.json").exists():
        write_fast_qwen3_artifact(model_dir, artifact_dir)

    store = FastQwen3Store(artifact_dir)
    model_config = store.manifest["model_config"]
    num_attention_heads = int(model_config["num_attention_heads"])
    num_kv_heads = int(model_config["num_key_value_heads"])
    if num_kv_heads != 8:
        raise ValueError("QwenPlaneAttentionCurrent currently requires 8 KV heads")
    head_dim = int(model_config["head_dim"])
    q_heads_per_group = num_attention_heads // num_kv_heads
    o_prefix = _linear_prefix(args.layer)
    o_spec = store.linear_spec(o_prefix)
    if o_spec.in_features != num_attention_heads * head_dim:
        raise ValueError("o_proj input does not match attention context width")
    max_o_patches = o_spec.padded_out_features // Q4NX_PATCH_OUT_ROWS
    start_patch = args.o_block * args.output_patches
    if args.o_block < 0 or start_patch + args.output_patches > max_o_patches:
        raise ValueError("o-block and output-patches exceed o_proj output size")
    o_weight, o_patch_views = _linear_weight_stream(
        store,
        o_prefix,
        start_patch,
        args.output_patches,
    )

    torch.manual_seed(550940579)
    q_current = torch.randn(
        num_kv_heads,
        (q_heads_per_group + 2) * head_dim,
        dtype=torch.bfloat16,
    )
    kv_plane = torch.randn(
        kv_plane_total_elements(args.packet_seq_len, head_dim),
        dtype=torch.bfloat16,
    )
    residual_block = torch.randn(
        args.output_patches * Q4NX_PATCH_OUT_ROWS,
        dtype=torch.bfloat16,
    )
    expected_context = plane_attention_current_reference(
        q_current,
        kv_plane,
        args.current_slot,
        args.attend_seq_len,
        args.packet_seq_len,
        q_heads_per_group,
    )
    expected_projected = torch.stack(
        [
            q4nx_output_patch_reference(
                expected_context.flatten(),
                patch,
                bf16_partial_accum=True,
            )
            for patch in o_patch_views
        ],
        dim=0,
    )
    expected = (
        expected_projected.flatten().to(torch.float32)
        + residual_block.to(torch.float32)
    ).to(torch.bfloat16)
    expected = expected.view(args.output_patches, Q4NX_PATCH_OUT_ROWS)

    context = AIEContext(build_dir=Path(args.build_dir))
    attention_op = QwenPlaneAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.current_slot,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        tile_size=args.tile_size,
        plane_fifo_depth=args.plane_fifo_depth,
        context=context,
    )
    o_proj_op = Q4NXFusedLinearResidualProjection(
        in_features=o_spec.in_features,
        output_patches=args.output_patches,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_plane_attention_o_projection_residual",
        runlist=[
            (attention_op, "q_current", "kv_plane", "attn_context"),
            (
                o_proj_op,
                "attn_context",
                "residual_block",
                "o_weight",
                "o_proj_residual",
            ),
        ],
        input_args=["q_current", "residual_block"],
        output_args=["o_proj_residual"],
        external_args={
            "kv_plane": ["kv_plane"],
            "o_weight": ["o_weight"],
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("q_current").torch_view().view_as(q_current)[:] = q_current
    fused.get_buffer("residual_block").torch_view()[:] = residual_block.flatten()
    fused.get_buffer("kv_plane").torch_view()[:] = kv_plane.flatten()
    fused.get_buffer("o_weight").torch_view()[:] = o_weight.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("kv_plane")
    fused.mark_buffer_dirty("o_weight")
    fused.kv_plane_buffer.to("npu")
    fused.o_weight_buffer.to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual = fused.get_buffer("o_proj_residual").torch_view().view_as(expected)
    error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_error = float(error.max().item())
    if max_error > args.abs_tol:
        raise AssertionError(
            f"plane_attention_o_proj_residual max_abs_error={max_error}"
        )

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "attend_seq_len": args.attend_seq_len,
                "build_dir": str(Path(args.build_dir).resolve()),
                "current_slot": args.current_slot,
                "head_dim": head_dim,
                "layer": args.layer,
                "o_block": args.o_block,
                "o_proj_residual_max_abs_error": max_error,
                "o_proj_residual_mean_abs_error": float(error.mean().item()),
                "o_proj_residual_shape": list(actual.shape),
                "o_weight_bytes": int(o_weight.numel()),
                "output_patches": args.output_patches,
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
                "q_heads_per_group": q_heads_per_group,
                "tile_size": args.tile_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
