#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and run the fast Qwen3 fused-QKV Q4NX patch operator."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from models.fast_qwen3.fast_packed_format import (
    FastQwen3Store,
    default_fast_dir,
    find_fast_dir,
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.operators.q4nx_fused_qkv_projection import Q4NXFusedQKVProjection
from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_PATCH_OUT_ROWS,
    q4nx_output_patch_reference,
)
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fast_qwen3 Q4NX QKV patch operator")
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--build-dir", default="build/fast_qwen3_qkv_patch_smoke")
    parser.add_argument("--trace-dir", default="build_trace/fast_qwen3_qkv_patch")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--in-features",
        type=int,
        default=0,
        help="Input K width to stream; default uses model hidden_size",
    )
    parser.add_argument("--output-patches", type=int, default=8)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.05)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--trace-size", type=int, default=0)
    return parser.parse_args()


def _artifact_dir(model_dir: Path, artifact_arg: str | None) -> Path:
    if artifact_arg is not None:
        return Path(artifact_arg).expanduser().resolve()
    existing = find_fast_dir(model_dir)
    return existing if existing is not None else default_fast_dir(model_dir)


def _linear_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.self_attn.{projection}_proj"


def _rms_norm_weight_name(layer: int) -> str:
    return f"model.layers.{layer}.input_layernorm.weight"


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    hidden_f32 = hidden.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    inv_rms = torch.rsqrt(hidden_f32.pow(2).mean() + epsilon)
    return (hidden_f32 * inv_rms * weight_f32).to(torch.bfloat16)


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16)
    return raw.numpy().tobytes()


def _trace_event_stats(trace_json: Path) -> dict[str, dict[str, float | int]]:
    if not trace_json.exists():
        return {}
    events = json.loads(trace_json.read_text())
    open_events: dict[str, list[float]] = {}
    durations: dict[str, list[float]] = {}
    for event in events:
        phase = event.get("ph")
        name = event.get("name")
        timestamp = event.get("ts")
        if not isinstance(name, str) or not isinstance(timestamp, int | float):
            continue
        if phase == "B":
            open_events.setdefault(name, []).append(float(timestamp))
        elif phase == "E" and name in open_events and open_events[name]:
            start = open_events[name].pop()
            durations.setdefault(name, []).append(float(timestamp) - start)

    return {
        name: {
            "count": len(values),
            "total_cycles": sum(values),
            "mean_cycles": statistics.fmean(values),
            "max_cycles": max(values),
        }
        for name, values in sorted(durations.items())
        if values
    }


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be non-negative")
    if args.timed_iters <= 0:
        raise ValueError("timed-iters must be positive")
    if args.trace_size < 0:
        raise ValueError("trace-size must be non-negative")

    model_dir = find_model_dir(args.model_dir)
    artifact_dir = _artifact_dir(model_dir, args.artifact_dir)
    if args.repack or not (artifact_dir / "manifest.json").exists():
        write_fast_qwen3_artifact(model_dir, artifact_dir)

    store = FastQwen3Store(artifact_dir)
    model_config = store.manifest["model_config"]
    in_features = (
        int(model_config["hidden_size"]) if args.in_features == 0 else args.in_features
    )
    rms_norm_epsilon = float(model_config.get("rms_norm_eps") or 1e-6)
    if in_features <= 0:
        raise ValueError("in-features must be positive")
    if in_features % 256 != 0:
        raise ValueError("in-features must be divisible by 256")
    if not 1 <= args.output_patches <= 8:
        raise ValueError("output-patches must be in [1, 8]")
    norm_weight = store.dense(_rms_norm_weight_name(args.layer)).to(torch.bfloat16)
    if norm_weight.numel() != in_features:
        raise ValueError(
            f"RMSNorm weight has {norm_weight.numel()} elements, expected {in_features}"
        )

    k_chunk_patch_bytes = 2 * Q4NX_CHUNK_BYTES
    k_chunks = in_features // 256
    first_patch_bytes = k_chunks * k_chunk_patch_bytes
    projection_patches = [
        [
            store.linear_bytes(_linear_prefix(args.layer, projection))[
                patch_idx * first_patch_bytes : (patch_idx + 1) * first_patch_bytes
            ]
            for patch_idx in range(args.output_patches)
        ]
        for projection in ("q", "k", "v")
    ]
    for projection_idx, projection in enumerate(("q", "k", "v")):
        for patch_idx, patch in enumerate(projection_patches[projection_idx]):
            if len(patch) != first_patch_bytes:
                raise ValueError(
                    f"{projection}_proj output patch {patch_idx} has {len(patch)} bytes, "
                    f"expected {first_patch_bytes}"
                )
    chunks: list[bytes] = []
    for patch_idx in range(args.output_patches):
        for k_idx in range(k_chunks):
            for projection_idx in range(3):
                patch = projection_patches[projection_idx][patch_idx]
                chunks.append(
                    bytes(
                        patch[
                            k_idx * k_chunk_patch_bytes : (k_idx + 1)
                            * k_chunk_patch_bytes
                        ]
                    )
                )
    qkv_stream = b"".join(chunks)
    qkv_weight = torch.from_numpy(
        np.frombuffer(qkv_stream, dtype=np.uint8).copy()
    )

    torch.manual_seed(1608560892)
    hidden = torch.randn(in_features, dtype=torch.bfloat16)
    normed_hidden = _rms_norm(hidden, norm_weight, rms_norm_epsilon)
    expected = torch.stack(
        [
            torch.stack(
                [
                    q4nx_output_patch_reference(
                        normed_hidden,
                        projection_patches[projection_idx][patch_idx],
                        bf16_partial_accum=True,
                    )
                    for projection_idx in range(3)
                ],
                dim=0,
            )
            for patch_idx in range(args.output_patches)
        ],
        dim=0,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    op = Q4NXFusedQKVProjection(
        in_features=in_features,
        output_patches=args.output_patches,
        rms_norm_epsilon=rms_norm_epsilon,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_qkv_patch",
        runlist=[(op, "hidden", "norm_weight", "qkv_weight", "qkv_out")],
        input_args=["hidden"],
        output_args=["qkv_out"],
        external_args={
            "norm_weight": ["norm_weight"],
            "qkv_weight": ["qkv_weight"],
        },
        compile_mode="full_elf_dynamic",
        trace_size=args.trace_size,
        trace_file=trace_dir / "qkv_patch.trace.txt",
        trace_json_file=trace_dir / "qkv_patch.trace.json",
        trace_op_index=0,
        trace_ddr_id=4,
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("hidden").torch_view()[:] = hidden.flatten()
    fused.get_buffer("norm_weight").torch_view()[:] = norm_weight.flatten()
    fused.get_buffer("qkv_weight").torch_view()[:] = qkv_weight.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("norm_weight")
    fused.mark_buffer_dirty("qkv_weight")
    fused.norm_weight_buffer.to("npu")
    fused.qkv_weight_buffer.to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual = (
        fused.get_buffer("qkv_out")
        .torch_view()
        .view(args.output_patches, 3, Q4NX_PATCH_OUT_ROWS)
    )
    abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            f"Q4NX fused QKV patch max_abs_error={max_abs_error} exceeds {args.abs_tol}"
        )

    trace_json = trace_dir / "qkv_patch.trace.json"
    trace_text = trace_dir / "qkv_patch.trace.txt"
    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "layer": args.layer,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "in_features": in_features,
                "k_chunks": k_chunks,
                "output_patches": args.output_patches,
                "output_shape": list(actual.shape),
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "qkv_patch_bytes": first_patch_bytes,
                "qkv_stream_bytes": int(qkv_weight.numel()),
                "rms_norm_epsilon": rms_norm_epsilon,
                "trace_error": fused.last_trace_error,
                "trace_event_stats": _trace_event_stats(trace_json)
                if args.trace_size > 0
                else {},
                "trace_json_file": str(trace_json.resolve()) if args.trace_size > 0 else None,
                "trace_summary": fused.last_trace_summary,
                "trace_text_file": str(trace_text.resolve()) if args.trace_size > 0 else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
