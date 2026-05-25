#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and run the fast Qwen3 Q4NX attention output projection operator."""

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
from models.fast_qwen3.operators.q4nx_fused_linear_projection import (
    Q4NXFusedLinearProjection,
)
from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_PATCH_OUT_ROWS,
    q4nx_output_patch_reference,
)
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fast_qwen3 Q4NX o_proj patch operator"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--build-dir", default="build/fast_qwen3_o_projection_smoke")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-patches", type=int, default=8)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.06)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
    return parser.parse_args()


def _artifact_dir(model_dir: Path, artifact_arg: str | None) -> Path:
    if artifact_arg is not None:
        return Path(artifact_arg).expanduser().resolve()
    existing = find_fast_dir(model_dir)
    return existing if existing is not None else default_fast_dir(model_dir)


def _linear_prefix(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj"


def _weight_stream(
    store: FastQwen3Store,
    prefix: str,
    output_patches: int,
) -> tuple[torch.Tensor, tuple[memoryview, ...]]:
    spec = store.linear_spec(prefix)
    if not 1 <= output_patches <= spec.padded_out_features // Q4NX_PATCH_OUT_ROWS:
        raise ValueError("output-patches exceeds o_proj output patch count")
    patch_k_chunk_bytes = 2 * Q4NX_CHUNK_BYTES
    k_chunks = spec.in_features // 256
    patch_bytes = k_chunks * patch_k_chunk_bytes
    patch_views = tuple(
        store.linear_bytes(prefix)[
            patch_idx * patch_bytes : (patch_idx + 1) * patch_bytes
        ]
        for patch_idx in range(output_patches)
    )
    chunks: list[bytes] = []
    for patch in patch_views:
        for chunk_idx in range(2):
            for k_idx in range(k_chunks):
                start = k_idx * patch_k_chunk_bytes + chunk_idx * Q4NX_CHUNK_BYTES
                chunks.append(bytes(patch[start : start + Q4NX_CHUNK_BYTES]))
    return torch.from_numpy(np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()), patch_views


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
    prefix = _linear_prefix(args.layer)
    spec = store.linear_spec(prefix)
    if spec.in_features % 256 != 0:
        raise ValueError("o_proj in_features must be divisible by 256")
    linear_weight, patch_views = _weight_stream(store, prefix, args.output_patches)

    torch.manual_seed(1608560892)
    attention_context = torch.randn(spec.in_features, dtype=torch.bfloat16)
    expected = torch.stack(
        [
            q4nx_output_patch_reference(
                attention_context,
                patch,
                bf16_partial_accum=True,
            )
            for patch in patch_views
        ],
        dim=0,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    op = Q4NXFusedLinearProjection(
        in_features=spec.in_features,
        output_patches=args.output_patches,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_o_projection",
        runlist=[(op, "attention_context", "linear_weight", "linear_out")],
        input_args=["attention_context"],
        output_args=["linear_out"],
        external_args={"linear_weight": ["linear_weight"]},
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("attention_context").torch_view()[:] = attention_context.flatten()
    fused.get_buffer("linear_weight").torch_view()[:] = linear_weight.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("linear_weight")
    fused.linear_weight_buffer.to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual = fused.get_buffer("linear_out").torch_view().view_as(expected)
    abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            f"o_proj max_abs_error={max_abs_error} exceeds {args.abs_tol}"
        )

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "in_features": spec.in_features,
                "layer": args.layer,
                "linear": prefix,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
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
                "weight_stream_bytes": int(linear_weight.numel()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
