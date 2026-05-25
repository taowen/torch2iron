#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and run fast Qwen3 current-aware attention with packet update."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

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
from models.fast_qwen3.operators.qwen_chunked_attention_current import (
    QwenChunkedAttentionCurrent,
)
from models.fast_qwen3.operators.qwen_current_kv_cache_write import (
    QwenCurrentKVCacheWrite,
)
from models.quantized_qwen3.model import find_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast_qwen3 current-aware attention with packet update"
    )
    parser.add_argument("model_dir", help="Qwen3 model directory or parent directory")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--build-dir", default="build/fast_qwen3_attention_current_smoke")
    parser.add_argument("--trace-dir", default="build_trace/fast_qwen3_attention_current")
    parser.add_argument("--packet-seq-len", type=int, default=128)
    parser.add_argument("--attend-seq-len", type=int, default=128)
    parser.add_argument("--current-slot", type=int, default=73)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--num-kv-groups",
        type=int,
        default=1,
        help="Number of KV groups to instantiate for this placement smoke",
    )
    parser.add_argument("--abs-tol", type=float, default=0.08)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--trace-size", type=int, default=0)
    parser.add_argument("--repack", action="store_true")
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
            packet[value_offset : value_offset + head_dim] = previous_values[group_idx, slot]
            packet[mask_offset] = torch.tensor(1.0, dtype=packet.dtype)


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
    num_attention_heads = int(model_config["num_attention_heads"])
    model_kv_groups = int(model_config["num_key_value_heads"])
    num_kv_groups = args.num_kv_groups
    if num_kv_groups <= 0 or num_kv_groups > model_kv_groups:
        raise ValueError(f"num-kv-groups must be in [1, {model_kv_groups}]")
    head_dim = int(model_config["head_dim"])
    q_heads_per_group = num_attention_heads // model_kv_groups
    packet_elements = decode_packet_elements(
        num_kv_groups,
        args.packet_seq_len,
        args.chunk_size,
        head_dim,
    )

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
    current_row = args.current_slot % args.chunk_size
    valid_mask = (
        torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        if current_row == 0
        else torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
    )
    packet = torch.zeros((packet_elements,), dtype=torch.bfloat16)
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
    expected_context, expected_packet = chunked_attention_update_reference(
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
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    op = QwenChunkedAttentionCurrent(
        packet_seq_len=args.packet_seq_len,
        attend_seq_len=args.attend_seq_len,
        current_slot=args.current_slot,
        num_kv_groups=num_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    cache_write_op = QwenCurrentKVCacheWrite(
        packet_seq_len=args.packet_seq_len,
        current_slot=args.current_slot,
        num_kv_groups=num_kv_groups,
        q_heads_per_group=q_heads_per_group,
        head_dim=head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    fused_op = FusedMLIROperator(
        name="fast_qwen3_attention_current",
        runlist=[
            (
                op,
                "q_current",
                "packet_cache",
                "attn_context",
            ),
            (
                cache_write_op,
                "q_current",
                "valid_mask",
                "packet_cache",
            ),
        ],
        input_args=["q_current", "valid_mask"],
        output_args=["attn_context"],
        external_args={"kv_cache": ["packet_cache"]},
        compile_mode="full_elf_dynamic",
        trace_size=args.trace_size,
        trace_file=trace_dir / "attention_current.trace.txt",
        trace_json_file=trace_dir / "attention_current.trace.json",
        trace_op_index=0,
        trace_ddr_id=5,
        context=context,
    ).compile()
    fused = fused_op.get_callable()
    fused.get_buffer("q_current").torch_view()[:] = q_current.flatten()
    fused.get_buffer("valid_mask").torch_view()[:] = valid_mask.flatten()
    fused.get_buffer("packet_cache").torch_view()[:] = packet.flatten()
    fused.mark_buffer_dirty("input")
    fused.mark_buffer_dirty("kv_cache")
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
        .view(num_kv_groups, q_heads_per_group, head_dim)
    )
    fused.kv_cache_buffer.to("cpu")
    actual_packet = fused.get_buffer("packet_cache").torch_view()
    context_error = (actual_context.to(torch.float32) - expected_context.to(torch.float32)).abs()
    packet_error = (actual_packet.to(torch.float32) - expected_packet.to(torch.float32)).abs()
    max_context_error = float(context_error.max().item())
    max_packet_error = float(packet_error.max().item())
    if max_context_error > args.abs_tol:
        raise AssertionError(
            f"attention max_abs_error={max_context_error} exceeds {args.abs_tol}"
        )
    if max_packet_error > 0.0:
        raise AssertionError(f"packet update max_abs_error={max_packet_error}")

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "build_dir": str(Path(args.build_dir).resolve()),
                "num_kv_groups": num_kv_groups,
                "q_heads_per_group": q_heads_per_group,
                "head_dim": head_dim,
                "packet_seq_len": args.packet_seq_len,
                "attend_seq_len": args.attend_seq_len,
                "current_slot": args.current_slot,
                "chunk_size": args.chunk_size,
                "output_shape": list(actual_context.shape),
                "max_abs_error": max_context_error,
                "mean_abs_error": float(context_error.mean().item()),
                "packet_update": "persisted_by_qwen_current_kv_cache_write",
                "packet_update_max_abs_error": max_packet_error,
                "profile": {
                    "mean_us": statistics.fmean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "timed_iters": args.timed_iters,
                    "warmup_iters": args.warmup_iters,
                },
                "trace_error": fused.last_trace_error,
                "trace_summary": fused.last_trace_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
