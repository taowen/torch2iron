#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run direct group-local q_current projection into attention."""

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

from models.fast_qwen3.attention_reference import (
    chunked_attention_update_reference,
    decode_packet_elements,
    decode_packet_slot_offsets,
)
from models.fast_qwen3.fast_packed_format import (
    FastQwen3Store,
    default_fast_dir,
    find_fast_dir,
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.operators.q4nx_fused_q_current_projection import (
    Q4NXFusedQCurrentProjection,
)
from models.fast_qwen3.operators.qwen_chunked_attention_current import (
    QwenChunkedAttentionCurrent,
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
        description="Run fast_qwen3 direct q_current projection through attention"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--build-dir", default="build/fast_qwen3_q_current_attention")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--num-kv-groups", type=int, default=1)
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=128)
    parser.add_argument("--current-slot", type=int, default=73)
    parser.add_argument("--chunk-size", type=int, default=64)
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


def _linear_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.self_attn.{projection}_proj"


def _rms_norm_weight_name(layer: int) -> str:
    return f"model.layers.{layer}.input_layernorm.weight"


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    hidden_f32 = hidden.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    inv_rms = torch.rsqrt(hidden_f32.pow(2).mean() + epsilon)
    return (hidden_f32 * inv_rms * weight_f32).to(torch.bfloat16)


def _fill_previous_packet_slots(
    packet: torch.Tensor,
    previous_keys: torch.Tensor,
    previous_values: torch.Tensor,
    valid_tokens: int,
    packet_seq_len: int,
    chunk_size: int,
) -> None:
    num_kv_groups = int(previous_keys.shape[0])
    head_dim = int(previous_keys.shape[2])
    for group_idx in range(num_kv_groups):
        for slot in range(valid_tokens):
            key_offset, value_offset, mask_offset = decode_packet_slot_offsets(
                group_idx,
                slot,
                packet_seq_len,
                chunk_size,
                head_dim,
            )
            packet[key_offset : key_offset + head_dim] = previous_keys[group_idx, slot]
            packet[value_offset : value_offset + head_dim] = previous_values[
                group_idx,
                slot,
            ]
            packet[mask_offset] = torch.tensor(1.0, dtype=packet.dtype)


def _q_current_weight_stream(
    store: FastQwen3Store,
    layer: int,
    in_features: int,
    group_index: int,
    q_heads_per_group: int,
    head_dim: int,
) -> tuple[torch.Tensor, tuple[memoryview, ...]]:
    k_chunk_patch_bytes = 2 * Q4NX_CHUNK_BYTES
    k_chunks = in_features // 256
    first_patch_bytes = k_chunks * k_chunk_patch_bytes
    projection_names = ("q", "k", "v")
    patch_plan = q_current_patch_plan(
        group_index,
        q_heads_per_group,
        head_dim,
        Q4NX_PATCH_OUT_ROWS,
    )
    patch_views = tuple(
        store.linear_bytes(_linear_prefix(layer, projection_names[projection_idx]))[
            patch_idx * first_patch_bytes : (patch_idx + 1) * first_patch_bytes
        ]
        for projection_idx, patch_idx in patch_plan
    )
    chunks: list[bytes] = []
    for patch in patch_views:
        for k_idx in range(k_chunks):
            chunks.append(
                bytes(
                    patch[
                        k_idx * k_chunk_patch_bytes : (k_idx + 1) * k_chunk_patch_bytes
                    ]
                )
            )
    return torch.from_numpy(np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()), patch_views


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
    model_kv_groups = int(model_config["num_key_value_heads"])
    group_indices = tuple(
        range(args.group_index, args.group_index + args.num_kv_groups)
    )
    if group_indices[-1] >= model_kv_groups:
        raise ValueError(
            f"group-index + num-kv-groups must be <= {model_kv_groups}"
        )
    q_heads_per_group = int(model_config["num_attention_heads"]) // model_kv_groups
    rms_norm_epsilon = float(model_config.get("rms_norm_eps") or 1e-6)

    q_current_weights: list[torch.Tensor] = []
    group_patch_views: list[tuple[memoryview, ...]] = []
    for group_index in group_indices:
        group_weight, patch_views = _q_current_weight_stream(
            store,
            args.layer,
            hidden_size,
            group_index,
            q_heads_per_group,
            head_dim,
        )
        q_current_weights.append(group_weight)
        group_patch_views.append(patch_views)
    q_current_weight = torch.cat(q_current_weights)
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
    q_elements_per_group = q_heads_per_group * head_dim
    queries = q_current_expected[:, :q_elements_per_group].view(
        args.num_kv_groups,
        q_heads_per_group,
        head_dim,
    )
    current_key = q_current_expected[
        :,
        q_elements_per_group : q_elements_per_group + head_dim
    ]
    current_value = q_current_expected[:, q_elements_per_group + head_dim :]
    packet = torch.zeros(
        (
            decode_packet_elements(
                args.num_kv_groups,
                args.packet_seq_len,
                args.chunk_size,
                head_dim,
            ),
        ),
        dtype=torch.bfloat16,
    )
    previous_keys = torch.randn(
        (args.num_kv_groups, args.current_slot, head_dim),
        dtype=torch.bfloat16,
    )
    previous_values = torch.randn(
        (args.num_kv_groups, args.current_slot, head_dim),
        dtype=torch.bfloat16,
    )
    _fill_previous_packet_slots(
        packet,
        previous_keys,
        previous_values,
        args.current_slot,
        args.packet_seq_len,
        args.chunk_size,
    )
    expected_context, _expected_packet = chunked_attention_update_reference(
        queries,
        current_key,
        current_value,
        packet,
        args.current_slot,
        args.attend_seq_len,
        args.packet_seq_len,
        args.chunk_size,
    )
    context = AIEContext(build_dir=Path(args.build_dir))
    q_current_op = Q4NXFusedQCurrentProjection(
        in_features=hidden_size,
        num_kv_groups=args.num_kv_groups,
        group_index=args.group_index,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        rms_norm_epsilon=rms_norm_epsilon,
        context=context,
    )
    attn_op = QwenChunkedAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.current_slot,
        num_kv_groups=args.num_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_q_current_attention",
        runlist=[
            (
                q_current_op,
                "hidden",
                "norm_weight",
                "q_current_weight",
                "q_current",
            ),
            (attn_op, "q_current", "packet_cache", "attn_context"),
        ],
        input_args=["hidden"],
        output_args=["attn_context"],
        external_args={
            "norm_weight": ["norm_weight"],
            "q_current_weight": ["q_current_weight"],
            "kv_cache": ["packet_cache"],
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("hidden").torch_view()[:] = hidden.flatten()
    fused.get_buffer("norm_weight").torch_view()[:] = norm_weight.flatten()
    fused.get_buffer("q_current_weight").torch_view()[:] = q_current_weight.flatten()
    fused.get_buffer("packet_cache").torch_view()[:] = packet.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("norm_weight")
    fused.mark_buffer_dirty("q_current_weight")
    fused.mark_buffer_dirty("kv_cache")
    fused.norm_weight_buffer.to("npu")
    fused.q_current_weight_buffer.to("npu")
    fused.kv_cache_buffer.to("npu")
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
        .view(args.num_kv_groups, q_heads_per_group, head_dim)
    )
    context_error = (
        actual_context.to(torch.float32) - expected_context.to(torch.float32)
    ).abs()
    max_context_error = float(context_error.max().item())
    if max_context_error > args.abs_tol:
        raise AssertionError(f"attention max_abs_error={max_context_error}")

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "layer": args.layer,
                "group_indices": list(group_indices),
                "q_heads_per_group": q_heads_per_group,
                "head_dim": head_dim,
                "q_current_shape": list(q_current_expected.shape),
                "context_shape": list(actual_context.shape),
                "q_current_weight_bytes": int(q_current_weight.numel()),
                "attention_max_abs_error": max_context_error,
                "attention_mean_abs_error": float(context_error.mean().item()),
                "packet_update": "covered_by_attention_current_smoke",
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
