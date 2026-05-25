#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and run direct group-local q_current projection."""

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
from models.fast_qwen3.operators.q4nx_fused_q_current_projection import (
    Q4NXFusedQCurrentProjection,
)
from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_PATCH_OUT_ROWS,
    q4nx_output_patch_reference,
)
from models.fast_qwen3.qkv_reference import q_current_patch_plan
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fast_qwen3 direct q_current projection operator"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--build-dir", default="build/fast_qwen3_q_current_smoke")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--num-kv-groups", type=int, default=1)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.05)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
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


def q_current_weight_stream(
    store: FastQwen3Store,
    layer: int,
    in_features: int,
    group_index: int,
    num_kv_groups: int,
    q_heads_per_group: int,
    head_dim: int,
) -> tuple[torch.Tensor, tuple[tuple[memoryview, ...], ...]]:
    k_chunk_patch_bytes = 2 * Q4NX_CHUNK_BYTES
    k_chunks = in_features // 256
    patch_bytes = k_chunks * k_chunk_patch_bytes
    projection_names = ("q", "k", "v")
    group_patch_views = tuple(
        tuple(
            store.linear_bytes(_linear_prefix(layer, projection_names[projection_idx]))[
                patch_idx * patch_bytes : (patch_idx + 1) * patch_bytes
            ]
            for projection_idx, patch_idx in q_current_patch_plan(
                group_idx,
                q_heads_per_group,
                head_dim,
                Q4NX_PATCH_OUT_ROWS,
            )
        )
        for group_idx in range(group_index, group_index + num_kv_groups)
    )
    chunks: list[bytes] = []
    for patch_views in group_patch_views:
        for patch in patch_views:
            for k_idx in range(k_chunks):
                chunks.append(
                    bytes(
                        patch[
                            k_idx * k_chunk_patch_bytes : (k_idx + 1)
                            * k_chunk_patch_bytes
                        ]
                    )
                )
    return (
        torch.from_numpy(np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()),
        group_patch_views,
    )


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be non-negative")
    if args.timed_iters <= 0:
        raise ValueError("timed-iters must be positive")
    if args.num_kv_groups <= 0:
        raise ValueError("num-kv-groups must be positive")

    model_dir = find_model_dir(args.model_dir)
    artifact_dir = _artifact_dir(model_dir, args.artifact_dir)
    if args.repack or not (artifact_dir / "manifest.json").exists():
        write_fast_qwen3_artifact(model_dir, artifact_dir)

    store = FastQwen3Store(artifact_dir)
    model_config = store.manifest["model_config"]
    hidden_size = int(model_config["hidden_size"])
    head_dim = int(model_config["head_dim"])
    num_kv_heads = int(model_config["num_key_value_heads"])
    group_indices = tuple(
        range(args.group_index, args.group_index + args.num_kv_groups)
    )
    if group_indices[-1] >= num_kv_heads:
        raise ValueError(f"group-index + num-kv-groups must be <= {num_kv_heads}")
    q_heads_per_group = int(model_config["num_attention_heads"]) // num_kv_heads
    rms_norm_epsilon = float(model_config.get("rms_norm_eps") or 1e-6)

    q_current_weight, group_patch_views = q_current_weight_stream(
        store,
        args.layer,
        hidden_size,
        args.group_index,
        args.num_kv_groups,
        q_heads_per_group,
        head_dim,
    )
    norm_weight = store.dense(_rms_norm_weight_name(args.layer)).to(torch.bfloat16)
    torch.manual_seed(1608560892)
    hidden = torch.randn(hidden_size, dtype=torch.bfloat16)
    normed_hidden = _rms_norm(hidden, norm_weight, rms_norm_epsilon)
    expected = torch.stack(
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

    context = AIEContext(build_dir=Path(args.build_dir))
    op = Q4NXFusedQCurrentProjection(
        in_features=hidden_size,
        num_kv_groups=args.num_kv_groups,
        group_index=args.group_index,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        rms_norm_epsilon=rms_norm_epsilon,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_q_current_projection",
        runlist=[(op, "hidden", "norm_weight", "q_current_weight", "q_current")],
        input_args=["hidden"],
        output_args=["q_current"],
        external_args={
            "norm_weight": ["norm_weight"],
            "q_current_weight": ["q_current_weight"],
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("hidden").torch_view()[:] = hidden.flatten()
    fused.get_buffer("norm_weight").torch_view()[:] = norm_weight.flatten()
    fused.get_buffer("q_current_weight").torch_view()[:] = q_current_weight.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("norm_weight")
    fused.mark_buffer_dirty("q_current_weight")
    fused.norm_weight_buffer.to("npu")
    fused.q_current_weight_buffer.to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual = fused.get_buffer("q_current").torch_view().view_as(expected)
    abs_error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    if max_abs_error > args.abs_tol:
        raise AssertionError(
            f"q_current projection max_abs_error={max_abs_error} exceeds {args.abs_tol}"
        )

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "group_indices": list(group_indices),
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "layer": args.layer,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "q_current_shape": list(actual.shape),
                "q_current_weight_bytes": int(q_current_weight.numel()),
                "q_heads_per_group": q_heads_per_group,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
