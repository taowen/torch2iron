#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run current-aware attention into o_proj plus residual add."""

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
from models.fast_qwen3.operators.q4nx_fused_linear_residual_projection import (
    Q4NXFusedLinearResidualProjection,
)
from models.fast_qwen3.operators.qwen_chunked_attention_current import (
    QwenChunkedAttentionCurrent,
)
from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_PATCH_OUT_ROWS,
    q4nx_output_patch_reference,
)
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 attention through o_proj plus residual add"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--build-dir",
        default="build/fast_qwen3_attention_o_projection_residual",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--o-block", type=int, default=0)
    parser.add_argument("--o-block-count", type=int, default=1)
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=128)
    parser.add_argument("--current-slot", type=int, default=73)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--repack", action="store_true")
    parser.add_argument("--abs-tol", type=float, default=0.10)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
    return parser.parse_args()


def _artifact_dir(model_dir: Path, artifact_arg: str | None) -> Path:
    if artifact_arg is not None:
        return Path(artifact_arg).expanduser().resolve()
    existing = find_fast_dir(model_dir)
    return existing if existing is not None else default_fast_dir(model_dir)


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


def _linear_prefix(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj"


def _linear_weight_stream(
    store: FastQwen3Store,
    prefix: str,
    start_patch: int,
    output_patches: int,
) -> tuple[torch.Tensor, tuple[memoryview, ...]]:
    spec = store.linear_spec(prefix)
    max_patches = spec.padded_out_features // Q4NX_PATCH_OUT_ROWS
    if (
        start_patch < 0
        or output_patches <= 0
        or start_patch + output_patches > max_patches
    ):
        raise ValueError("requested output patch range is outside linear weight")
    patch_k_chunk_bytes = 2 * Q4NX_CHUNK_BYTES
    k_chunks = spec.in_features // 256
    patch_bytes = k_chunks * patch_k_chunk_bytes
    patch_views = tuple(
        store.linear_bytes(prefix)[
            patch_idx * patch_bytes : (patch_idx + 1) * patch_bytes
        ]
        for patch_idx in range(start_patch, start_patch + output_patches)
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

    model_dir = find_model_dir(args.model_dir)
    artifact_dir = _artifact_dir(model_dir, args.artifact_dir)
    if args.repack or not (artifact_dir / "manifest.json").exists():
        write_fast_qwen3_artifact(model_dir, artifact_dir)

    store = FastQwen3Store(artifact_dir)
    model_config = store.manifest["model_config"]
    num_attention_heads = int(model_config["num_attention_heads"])
    num_kv_groups = int(model_config["num_key_value_heads"])
    head_dim = int(model_config["head_dim"])
    q_heads_per_group = num_attention_heads // num_kv_groups
    o_block_patches = 8
    o_prefix = _linear_prefix(args.layer)
    o_spec = store.linear_spec(o_prefix)
    if o_spec.in_features != num_attention_heads * head_dim:
        raise ValueError("o_proj input does not match attention context width")
    max_o_blocks = o_spec.padded_out_features // (
        o_block_patches * Q4NX_PATCH_OUT_ROWS
    )
    if args.o_block_count <= 0:
        raise ValueError("o-block-count must be positive")
    if args.o_block < 0 or args.o_block + args.o_block_count > max_o_blocks:
        raise ValueError(
            f"o-block + o-block-count must be inside [0, {max_o_blocks}]"
        )
    o_weights: list[torch.Tensor] = []
    o_patch_views_by_block: list[tuple[memoryview, ...]] = []
    for block_offset in range(args.o_block_count):
        o_weight, o_patch_views = _linear_weight_stream(
            store,
            o_prefix,
            (args.o_block + block_offset) * o_block_patches,
            o_block_patches,
        )
        o_weights.append(o_weight)
        o_patch_views_by_block.append(o_patch_views)

    torch.manual_seed(1608560892)
    queries = torch.randn(
        (num_kv_groups, q_heads_per_group, head_dim),
        dtype=torch.bfloat16,
    )
    current_key = torch.randn((num_kv_groups, head_dim), dtype=torch.bfloat16)
    current_value = torch.randn((num_kv_groups, head_dim), dtype=torch.bfloat16)
    q_current = torch.cat(
        (
            queries.reshape(num_kv_groups, q_heads_per_group * head_dim),
            current_key,
            current_value,
        ),
        dim=1,
    )
    packet = torch.zeros(
        (
            decode_packet_elements(
                num_kv_groups,
                args.packet_seq_len,
                args.chunk_size,
                head_dim,
            ),
        ),
        dtype=torch.bfloat16,
    )
    previous_keys = torch.randn(
        (num_kv_groups, args.current_slot, head_dim),
        dtype=torch.bfloat16,
    )
    previous_values = torch.randn(
        (num_kv_groups, args.current_slot, head_dim),
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
    residual_blocks = [
        torch.randn(
            o_block_patches * Q4NX_PATCH_OUT_ROWS,
            dtype=torch.bfloat16,
        )
        for _ in range(args.o_block_count)
    ]
    expected_blocks: list[torch.Tensor] = []
    for block_offset, o_patch_views in enumerate(o_patch_views_by_block):
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
        expected_block = (
            expected_projected.flatten().to(torch.float32)
            + residual_blocks[block_offset].to(torch.float32)
        ).to(torch.bfloat16)
        expected_blocks.append(expected_block.view(o_block_patches, Q4NX_PATCH_OUT_ROWS))

    context = AIEContext(build_dir=Path(args.build_dir))
    attn_op = QwenChunkedAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.current_slot,
        num_kv_groups=num_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    o_proj_op = Q4NXFusedLinearResidualProjection(
        in_features=o_spec.in_features,
        output_patches=o_block_patches,
        context=context,
    )
    o_runlist = [
        (
            o_proj_op,
            "attn_context",
            f"residual_block_{block_offset}",
            f"o_weight_{block_offset}",
            f"o_proj_residual_{block_offset}",
        )
        for block_offset in range(args.o_block_count)
    ]
    fused_op = FusedMLIROperator(
        name="fast_qwen3_attention_o_projection_residual",
        runlist=[
            (attn_op, "q_current", "packet_cache", "attn_context"),
            *o_runlist,
        ],
        input_args=[
            "q_current",
            *(f"residual_block_{idx}" for idx in range(args.o_block_count)),
        ],
        output_args=[
            f"o_proj_residual_{idx}" for idx in range(args.o_block_count)
        ],
        external_args={
            "kv_cache": ["packet_cache"],
            **{
                f"o_weight_{idx}": [f"o_weight_{idx}"]
                for idx in range(args.o_block_count)
            },
        },
        compile_mode="full_elf_dynamic",
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("q_current").torch_view()[:] = q_current.flatten()
    for block_offset, residual_block in enumerate(residual_blocks):
        fused.get_buffer(f"residual_block_{block_offset}").torch_view()[
            :
        ] = residual_block.flatten()
    fused.get_buffer("packet_cache").torch_view()[:] = packet.flatten()
    for block_offset, o_weight in enumerate(o_weights):
        fused.get_buffer(f"o_weight_{block_offset}").torch_view()[:] = o_weight.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("kv_cache")
    for block_offset in range(args.o_block_count):
        fused.mark_buffer_dirty(f"o_weight_{block_offset}")
    fused.kv_cache_buffer.to("npu")
    for block_offset in range(args.o_block_count):
        getattr(fused, f"o_weight_{block_offset}_buffer").to("npu")
    fused.output_buffer.to("npu")

    for _ in range(args.warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(args.timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    actual_blocks = [
        fused.get_buffer(f"o_proj_residual_{idx}")
        .torch_view()
        .view_as(expected_blocks[idx])
        for idx in range(args.o_block_count)
    ]
    actual_o = torch.cat(actual_blocks, dim=0)
    expected_o = torch.cat(expected_blocks, dim=0)
    o_error = (actual_o.to(torch.float32) - expected_o.to(torch.float32)).abs()
    max_o_error = float(o_error.max().item())
    if max_o_error > args.abs_tol:
        raise AssertionError(f"attention_o_proj_residual max_abs_error={max_o_error}")

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "head_dim": head_dim,
                "layer": args.layer,
                "num_kv_groups": num_kv_groups,
                "o_block": args.o_block,
                "o_block_count": args.o_block_count,
                "o_proj_residual_max_abs_error": max_o_error,
                "o_proj_residual_mean_abs_error": float(o_error.mean().item()),
                "o_proj_residual_shape": list(actual_o.shape),
                "o_weight_bytes": int(sum(weight.numel() for weight in o_weights)),
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "q_heads_per_group": q_heads_per_group,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
