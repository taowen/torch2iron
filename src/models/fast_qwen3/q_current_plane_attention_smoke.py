#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run packed-Q4 q_current projection into FastFlowLM-style plane attention."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from models.fast_qwen3.fast_packed_format import (
    FastQwen3Store,
    default_fast_dir,
    find_fast_dir,
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.kv_plane_reference import (
    kv_plane_total_elements,
    plane_attention_current_reference,
)
from models.fast_qwen3.operators.q4nx_fused_q_current_projection import (
    Q4NXFusedQCurrentProjection,
)
from models.fast_qwen3.operators.qwen_plane_attention_current import (
    QwenPlaneAttentionCurrent,
)
from models.fast_qwen3.q_current_operator_smoke import (
    _rms_norm,
    _rms_norm_weight_name,
    q_current_weight_stream,
)
from models.fast_qwen3.q4nx_layout import q4nx_output_patch_reference
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 q_current projection into plane-layout attention"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--build-dir",
        default="build/fast_qwen3_q_current_plane_attention",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=64)
    parser.add_argument("--current-slot", type=int, default=63)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--plane-fifo-depth", type=int, default=2)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.08)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
    return parser.parse_args()


def _artifact_dir(model_dir: Path, artifact_arg: str | None) -> Path:
    if artifact_arg is not None:
        return Path(artifact_arg).expanduser().resolve()
    existing = find_fast_dir(model_dir)
    return existing if existing is not None else default_fast_dir(model_dir)


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be non-negative")
    if args.timed_iters <= 0:
        raise ValueError("timed-iters must be positive")

    model_dir = find_model_dir(args.model_dir)
    artifact_dir = _artifact_dir(model_dir, args.artifact_dir)
    if args.repack or not (artifact_dir / "manifest.json").exists():
        write_fast_qwen3_artifact(model_dir, artifact_dir)

    store = FastQwen3Store(artifact_dir)
    model_config = store.manifest["model_config"]
    hidden_size = int(model_config["hidden_size"])
    head_dim = int(model_config["head_dim"])
    num_kv_heads = int(model_config["num_key_value_heads"])
    if num_kv_heads != 8:
        raise ValueError("QwenPlaneAttentionCurrent currently requires 8 KV heads")
    q_heads_per_group = int(model_config["num_attention_heads"]) // num_kv_heads
    rms_norm_epsilon = float(model_config.get("rms_norm_eps") or 1e-6)

    q_current_weight, group_patch_views = q_current_weight_stream(
        store,
        args.layer,
        hidden_size,
        0,
        num_kv_heads,
        q_heads_per_group,
        head_dim,
    )
    norm_weight = store.dense(_rms_norm_weight_name(args.layer)).to(torch.bfloat16)
    torch.manual_seed(1608560892)
    hidden = torch.randn(hidden_size, dtype=torch.bfloat16)
    normed_hidden = _rms_norm(hidden, norm_weight, rms_norm_epsilon)
    q_current_expected = torch.stack(
        [
            torch.stack(
                [
                    q4nx_output_patch_reference(
                        normed_hidden,
                        patch,
                        bf16_partial_accum=True,
                    )
                    for patch in patch_views
                ],
                dim=0,
            ).flatten()
            for patch_views in group_patch_views
        ],
        dim=0,
    )

    kv_plane = torch.randn(
        kv_plane_total_elements(args.packet_seq_len, head_dim),
        dtype=torch.bfloat16,
    )
    expected_context = plane_attention_current_reference(
        q_current_expected,
        kv_plane,
        args.current_slot,
        args.attend_seq_len,
        args.packet_seq_len,
        q_heads_per_group,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    q_current_op = Q4NXFusedQCurrentProjection(
        in_features=hidden_size,
        num_kv_groups=num_kv_heads,
        group_index=0,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        rms_norm_epsilon=rms_norm_epsilon,
        context=context,
    )
    plane_attention_op = QwenPlaneAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.current_slot,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        tile_size=args.tile_size,
        plane_fifo_depth=args.plane_fifo_depth,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_q_current_plane_attention",
        runlist=[
            (
                q_current_op,
                "hidden",
                "norm_weight",
                "q_current_weight",
                "q_current",
            ),
            (plane_attention_op, "q_current", "kv_plane", "attn_context"),
        ],
        input_args=["hidden"],
        output_args=["attn_context"],
        external_args={
            "norm_weight": ["norm_weight"],
            "q_current_weight": ["q_current_weight"],
            "kv_plane": ["kv_plane"],
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("hidden").torch_view()[:] = hidden.flatten()
    fused.get_buffer("norm_weight").torch_view()[:] = norm_weight.flatten()
    fused.get_buffer("q_current_weight").torch_view()[:] = q_current_weight.flatten()
    fused.get_buffer("kv_plane").torch_view()[:] = kv_plane.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("norm_weight")
    fused.mark_buffer_dirty("q_current_weight")
    fused.mark_buffer_dirty("kv_plane")
    fused.norm_weight_buffer.to("npu")
    fused.q_current_weight_buffer.to("npu")
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
        fused.get_buffer("attn_context")
        .torch_view()
        .view(num_kv_heads, q_heads_per_group, head_dim)
    )
    context_error = (
        actual_context.to(torch.float32) - expected_context.to(torch.float32)
    ).abs()
    max_context_error = float(context_error.max().item())
    if max_context_error > args.abs_tol:
        raise AssertionError(f"plane attention max_abs_error={max_context_error}")

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "attention_max_abs_error": max_context_error,
                "attention_mean_abs_error": float(context_error.mean().item()),
                "attend_seq_len": args.attend_seq_len,
                "build_dir": str(Path(args.build_dir).resolve()),
                "context_shape": list(actual_context.shape),
                "current_slot": args.current_slot,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "layer": args.layer,
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
                "q_current_shape": list(q_current_expected.shape),
                "q_current_weight_bytes": int(q_current_weight.numel()),
                "q_heads_per_group": q_heads_per_group,
                "tile_size": args.tile_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
