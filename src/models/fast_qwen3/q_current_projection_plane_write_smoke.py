#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the first Qwen3 layer-fused slice.

This validates packed-Q4 q_current projection plus in-place four-plane K/V cache
write through ``Qwen3LayerFusedMLIROperator``.
"""

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
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.kv_plane_reference import (
    kv_plane_group_offsets,
    kv_plane_total_elements,
    update_q_current_in_kv_plane,
)
from models.fast_qwen3.operators.qwen3_layer_fused import Qwen3LayerFusedMLIROperator
from models.fast_qwen3.q_current_operator_smoke import (
    _rms_norm_weight_name,
    q_current_weight_stream,
)
from models.fast_qwen3.q_current_plane_write_attention_smoke import (
    _artifact_dir,
    _q_current_reference,
)
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 q_current projection with plane-cache write"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--build-dir",
        default="build/fast_qwen3_q_current_projection_plane_write",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--current-slot", type=int, default=0)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.08)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=3)
    return parser.parse_args()


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
        raise ValueError("Qwen plane cache currently requires 8 KV heads")
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
    torch.manual_seed(321946079)
    hidden = torch.randn(hidden_size, dtype=torch.bfloat16)
    kv_plane = torch.randn(
        kv_plane_total_elements(args.packet_seq_len, head_dim),
        dtype=torch.bfloat16,
    )
    expected_q_current = _q_current_reference(
        hidden,
        norm_weight,
        rms_norm_epsilon,
        group_patch_views,
    )
    expected_plane = kv_plane.clone()
    update_q_current_in_kv_plane(
        expected_q_current,
        expected_plane,
        args.current_slot,
        args.packet_seq_len,
        q_heads_per_group,
    )

    context = AIEContext(build_dir=Path(args.build_dir))
    op = Qwen3LayerFusedMLIROperator(
        in_features=hidden_size,
        num_kv_groups=num_kv_heads,
        group_index=0,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        rms_norm_epsilon=rms_norm_epsilon,
        packet_seq_len=args.packet_seq_len,
        current_slot=args.current_slot,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_qwen3_layer_fused_projection_write",
        runlist=[
            (
                op,
                "hidden",
                "norm_weight",
                "q_current_weight",
                "q_current",
                "kv_plane",
            )
        ],
        input_args=["hidden"],
        output_args=["q_current"],
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
    fused.get_buffer("kv_plane").torch_view().view_as(kv_plane)[:] = kv_plane
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

    actual_q_current = (
        fused.get_buffer("q_current").torch_view().view_as(expected_q_current)
    )
    fused.kv_plane_buffer.to("cpu")
    actual_plane = fused.get_buffer("kv_plane").torch_view().view_as(expected_plane)
    q_error = (
        actual_q_current.to(torch.float32) - expected_q_current.to(torch.float32)
    ).abs()
    plane_error = (
        actual_plane.to(torch.float32) - expected_plane.to(torch.float32)
    ).abs()
    written_mask = torch.zeros_like(plane_error, dtype=torch.bool)
    for group_idx in range(num_kv_heads):
        key_offset, value_offset = kv_plane_group_offsets(
            group_idx,
            args.current_slot,
            args.packet_seq_len,
            head_dim,
        )
        written_mask[key_offset : key_offset + head_dim] = True
        written_mask[value_offset : value_offset + head_dim] = True
    preserved_plane_error = plane_error[~written_mask]
    written_plane_error = plane_error[written_mask]
    max_q_error = float(q_error.max().item())
    max_preserved_plane_error = float(preserved_plane_error.max().item())
    max_written_plane_error = float(written_plane_error.max().item())
    if max_q_error > args.abs_tol:
        raise AssertionError(f"q_current max_abs_error={max_q_error}")
    if max_preserved_plane_error != 0.0:
        raise AssertionError(
            f"kv_plane preserved max_abs_error={max_preserved_plane_error}"
        )
    if max_written_plane_error > args.abs_tol:
        raise AssertionError(
            f"kv_plane written-row max_abs_error={max_written_plane_error}"
        )

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "current_slot": args.current_slot,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "layer": args.layer,
                "packet_seq_len": args.packet_seq_len,
                "plane_preserved_max_abs_error": max_preserved_plane_error,
                "plane_written_max_abs_error": max_written_plane_error,
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "q_current_max_abs_error": max_q_error,
                "q_current_mean_abs_error": float(q_error.mean().item()),
                "q_current_shape": list(actual_q_current.shape),
                "q_current_weight_bytes": int(q_current_weight.numel()),
                "q_heads_per_group": q_heads_per_group,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
