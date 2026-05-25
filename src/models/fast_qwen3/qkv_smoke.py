#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke check for the fast Qwen3 Q4NX Q/K/V projection artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.quantized_qwen3.model import find_model_dir
from models.fast_qwen3.fast_packed_format import (
    FastQwen3Store,
    default_fast_dir,
    find_fast_dir,
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.operators.q4nx_fused_qkv_projection import Q4NXFusedQKVProjection
from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_IN_CHUNK,
    assert_layout_constants,
    q4nx_patch_reference,
)
from models.fast_qwen3.qkv_reference import fused_qkv_reference, qkv_output_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast_qwen3 fused-QKV reference smoke check")
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None, help="Existing or output fast_qwen3 artifact directory")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--repack", action="store_true")
    return parser.parse_args()


def _artifact_dir(model_dir: Path, artifact_arg: str | None) -> Path:
    if artifact_arg is not None:
        return Path(artifact_arg).expanduser().resolve()
    existing = find_fast_dir(model_dir)
    return existing if existing is not None else default_fast_dir(model_dir)


def main() -> None:
    args = parse_args()
    assert_layout_constants()
    model_dir = find_model_dir(args.model_dir)
    artifact_dir = _artifact_dir(model_dir, args.artifact_dir)
    if args.repack or not (artifact_dir / "manifest.json").exists():
        write_fast_qwen3_artifact(model_dir, artifact_dir)

    store = FastQwen3Store(artifact_dir)
    config = store.manifest["model_config"]
    hidden_size = int(config["hidden_size"])

    torch.manual_seed(1608560892)
    hidden = torch.randn(args.rows, hidden_size, dtype=torch.bfloat16)
    output = fused_qkv_reference(store, args.layer, hidden)
    q_prefix = f"model.layers.{args.layer}.self_attn.q_proj"
    k_prefix = f"model.layers.{args.layer}.self_attn.k_proj"
    v_prefix = f"model.layers.{args.layer}.self_attn.v_proj"
    first_patch_bytes = 2 * Q4NX_CHUNK_BYTES
    hidden_patch = hidden[:, :Q4NX_IN_CHUNK]
    q_patch = q4nx_patch_reference(
        hidden_patch,
        store.linear_bytes(q_prefix)[:first_patch_bytes],
    )
    k_patch = q4nx_patch_reference(
        hidden_patch,
        store.linear_bytes(k_prefix)[:first_patch_bytes],
    )
    v_patch = q4nx_patch_reference(
        hidden_patch,
        store.linear_bytes(v_prefix)[:first_patch_bytes],
    )
    operator = Q4NXFusedQKVProjection(
        in_features=hidden_size,
        rms_norm_epsilon=float(config.get("rms_norm_eps") or 1e-6),
    )

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "layer": args.layer,
                "operator_arg_count": len(operator.get_arg_spec()),
                "operator_output_patches": operator.output_patches,
                "operator_patch_bytes": operator.patch_bytes,
                "operator_qkv_patch_stream_bytes": operator.qkv_patch_stream_bytes,
                "rows": args.rows,
                "shapes": qkv_output_shapes(output),
                "first_patch_shapes": {
                    "query": list(q_patch.shape),
                    "key": list(k_patch.shape),
                    "value": list(v_patch.shape),
                },
                "query_mean_abs": float(output.query.to(torch.float32).abs().mean().item()),
                "key_mean_abs": float(output.key.to(torch.float32).abs().mean().item()),
                "value_mean_abs": float(output.value.to(torch.float32).abs().mean().item()),
                "query_first_patch_mean_abs": float(
                    q_patch.to(torch.float32).abs().mean().item()
                ),
                "key_first_patch_mean_abs": float(
                    k_patch.to(torch.float32).abs().mean().item()
                ),
                "value_first_patch_mean_abs": float(
                    v_patch.to(torch.float32).abs().mean().item()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
