#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run QKV projection, q_current assembly, attention, and cache write together."""

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
    write_decode_packet_slot,
)
from models.fast_qwen3.fast_packed_format import (
    FastQwen3Store,
    default_fast_dir,
    find_fast_dir,
    write_fast_qwen3_artifact,
)
from models.fast_qwen3.operators.q4nx_fused_qkv_projection import Q4NXFusedQKVProjection
from models.fast_qwen3.operators.qwen_chunked_attention_current import (
    QwenChunkedAttentionCurrent,
)
from models.fast_qwen3.operators.qwen_current_kv_cache_write import (
    QwenCurrentKVCacheWrite,
)
from models.fast_qwen3.operators.qwen_qkv_to_q_current import QwenQKVToQCurrent
from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_PATCH_OUT_ROWS,
    q4nx_output_patch_reference,
)
from models.fast_qwen3.qkv_reference import assemble_q_current_from_qkv_patches
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 QKV projection through current attention"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--build-dir", default="build/fast_qwen3_qkv_attention_current")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=128)
    parser.add_argument("--current-slot", type=int, default=73)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output-patches", type=int, default=8)
    parser.add_argument("--num-kv-groups", type=int, default=2)
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
            packet[value_offset : value_offset + head_dim] = previous_values[group_idx, slot]
            packet[mask_offset] = torch.tensor(1.0, dtype=packet.dtype)


def _qkv_weight_stream(
    store: FastQwen3Store,
    layer: int,
    in_features: int,
    output_patches: int,
) -> tuple[torch.Tensor, list[list[memoryview]]]:
    k_chunk_patch_bytes = 2 * Q4NX_CHUNK_BYTES
    k_chunks = in_features // 256
    first_patch_bytes = k_chunks * k_chunk_patch_bytes
    projection_patches = [
        [
            store.linear_bytes(_linear_prefix(layer, projection))[
                patch_idx * first_patch_bytes : (patch_idx + 1) * first_patch_bytes
            ]
            for patch_idx in range(output_patches)
        ]
        for projection in ("q", "k", "v")
    ]
    chunks: list[bytes] = []
    for patch_idx in range(output_patches):
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
    return torch.from_numpy(np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()), projection_patches


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
    model_kv_groups = int(model_config["num_key_value_heads"])
    q_heads_per_group = int(model_config["num_attention_heads"]) // model_kv_groups
    rms_norm_epsilon = float(model_config.get("rms_norm_eps") or 1e-6)
    if args.num_kv_groups > model_kv_groups:
        raise ValueError(f"num-kv-groups must be <= {model_kv_groups}")

    qkv_weight, projection_patches = _qkv_weight_stream(
        store,
        args.layer,
        hidden_size,
        args.output_patches,
    )
    norm_weight = store.dense(_rms_norm_weight_name(args.layer)).to(torch.bfloat16)
    torch.manual_seed(1608560892)
    hidden = torch.randn(hidden_size, dtype=torch.bfloat16)
    normed_hidden = _rms_norm(hidden, norm_weight, rms_norm_epsilon)
    qkv_expected = torch.stack(
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
    q_current_expected = assemble_q_current_from_qkv_patches(
        qkv_expected,
        args.num_kv_groups,
        q_heads_per_group,
        head_dim,
    )
    q_elements_per_group = q_heads_per_group * head_dim
    queries = q_current_expected[:, :q_elements_per_group].reshape(
        args.num_kv_groups,
        q_heads_per_group,
        head_dim,
    )
    current_key = q_current_expected[
        :,
        q_elements_per_group : q_elements_per_group + head_dim,
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
    current_row = args.current_slot % args.chunk_size
    valid_mask = (
        torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        if current_row == 0
        else torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
    )
    context = AIEContext(build_dir=Path(args.build_dir))
    qkv_op = Q4NXFusedQKVProjection(
        in_features=hidden_size,
        output_patches=args.output_patches,
        rms_norm_epsilon=rms_norm_epsilon,
        context=context,
    )
    assemble_op = QwenQKVToQCurrent(
        qkv_output_patches=args.output_patches,
        num_kv_groups=args.num_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
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
    cache_write_op = QwenCurrentKVCacheWrite(
        packet_seq_len=args.packet_seq_len,
        current_slot=args.current_slot,
        num_kv_groups=args.num_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_qkv_attention_current",
        runlist=[
            (qkv_op, "hidden", "norm_weight", "qkv_weight", "qkv_out"),
            (assemble_op, "qkv_out", "q_current"),
            (attn_op, "q_current", "packet_cache", "attn_context"),
            (cache_write_op, "q_current", "valid_mask", "packet_cache"),
        ],
        input_args=["hidden", "valid_mask"],
        output_args=["q_current", "attn_context"],
        external_args={
            "norm_weight": ["norm_weight"],
            "qkv_weight": ["qkv_weight"],
            "kv_cache": ["packet_cache"],
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("hidden").torch_view()[:] = hidden.flatten()
    fused.get_buffer("valid_mask").torch_view()[:] = valid_mask.flatten()
    fused.get_buffer("norm_weight").torch_view()[:] = norm_weight.flatten()
    fused.get_buffer("qkv_weight").torch_view()[:] = qkv_weight.flatten()
    fused.get_buffer("packet_cache").torch_view()[:] = packet.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("norm_weight")
    fused.mark_buffer_dirty("qkv_weight")
    fused.mark_buffer_dirty("kv_cache")
    fused.norm_weight_buffer.to("npu")
    fused.qkv_weight_buffer.to("npu")
    fused.kv_cache_buffer.to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual_q_current = fused.get_buffer("q_current").torch_view().view_as(
        q_current_expected
    )
    actual_context = (
        fused.get_buffer("attn_context")
        .torch_view()
        .view(args.num_kv_groups, q_heads_per_group, head_dim)
    )
    fused.kv_cache_buffer.to("cpu")
    actual_packet = fused.get_buffer("packet_cache").torch_view()
    actual_key = actual_q_current[
        :,
        q_elements_per_group : q_elements_per_group + head_dim,
    ]
    actual_value = actual_q_current[:, q_elements_per_group + head_dim :]
    expected_packet_from_actual = write_decode_packet_slot(
        packet,
        actual_key,
        actual_value,
        args.current_slot,
        args.packet_seq_len,
        args.chunk_size,
    )
    q_current_error = (
        actual_q_current.to(torch.float32) - q_current_expected.to(torch.float32)
    ).abs()
    context_error = (
        actual_context.to(torch.float32) - expected_context.to(torch.float32)
    ).abs()
    packet_error = (
        actual_packet.to(torch.float32) - expected_packet_from_actual.to(torch.float32)
    ).abs()
    max_context_error = float(context_error.max().item())
    max_q_current_error = float(q_current_error.max().item())
    max_packet_error = float(packet_error.max().item())
    if max_q_current_error > args.abs_tol:
        raise AssertionError(f"q_current max_abs_error={max_q_current_error}")
    if max_context_error > args.abs_tol:
        raise AssertionError(f"attention max_abs_error={max_context_error}")
    if max_packet_error > 0.0:
        raise AssertionError(f"packet update max_abs_error={max_packet_error}")

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "layer": args.layer,
                "output_patches": args.output_patches,
                "num_kv_groups": args.num_kv_groups,
                "q_heads_per_group": q_heads_per_group,
                "head_dim": head_dim,
                "q_current_shape": list(actual_q_current.shape),
                "context_shape": list(actual_context.shape),
                "q_current_max_abs_error": max_q_current_error,
                "attention_max_abs_error": max_context_error,
                "attention_mean_abs_error": float(context_error.mean().item()),
                "packet_update_max_abs_error": max_packet_error,
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
