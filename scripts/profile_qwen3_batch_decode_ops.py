#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile Qwen3 batch decode operators with real decode shapes.

This intentionally profiles the individual operators instead of tuning fused
ELF tile parameters blindly.  The fused decode path currently submits all
RunOps as one XRT kernel, so host-visible timing is per full ELF.  One-op
full-ELF runs give a repeatable signal for which phase deserves a rewrite.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from torch2iron.operators import (
    ElementwiseAdd,
    GEMM,
    LlamaChunkedAttention,
    RMSNorm,
    RoPE,
    SiLUMul,
)


@dataclass(frozen=True)
class QwenShapeConfig:
    vocab_size: int
    emb_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_groups: int
    head_dim: int

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_groups * self.head_dim

    @property
    def q_heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_groups


def _load_config(model_dir: Path) -> QwenShapeConfig:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    hf_config = json.loads(config_path.read_text())
    return QwenShapeConfig(
        vocab_size=int(hf_config["vocab_size"]),
        emb_dim=int(hf_config["hidden_size"]),
        hidden_dim=int(hf_config["intermediate_size"]),
        n_heads=int(hf_config["num_attention_heads"]),
        n_kv_groups=int(hf_config["num_key_value_heads"]),
        head_dim=int(
            hf_config.get(
                "head_dim",
                int(hf_config["hidden_size"]) // int(hf_config["num_attention_heads"]),
            )
        ),
    )


def _bf16_random(shape) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16).contiguous()


def _make_attention_packet(op: LlamaChunkedAttention, valid_tokens: int) -> torch.Tensor:
    packet = torch.zeros((op.packed_elements,), dtype=torch.bfloat16)
    valid_tokens = min(valid_tokens, op.max_seq_len)
    for group_idx in range(op.num_kv_groups):
        group_base = group_idx * op.packed_elements_per_group
        for slot in range(valid_tokens):
            chunk_idx = slot // op.chunk_size
            row = slot % op.chunk_size
            chunk_base = group_base + chunk_idx * op.packed_chunk_elements
            k_offset = chunk_base + row * op.head_dim
            v_offset = chunk_base + op.chunk_size * op.head_dim + row * op.head_dim
            mask_offset = chunk_base + 2 * op.chunk_size * op.head_dim + row
            packet[k_offset : k_offset + op.head_dim] = _bf16_random((op.head_dim,))
            packet[v_offset : v_offset + op.head_dim] = _bf16_random((op.head_dim,))
            packet[mask_offset] = 1.0
    return packet


def _input_for_op(op, spec_idx: int, shape, *, valid_tokens: int) -> torch.Tensor:
    if isinstance(op, LlamaChunkedAttention) and spec_idx == 1:
        return _make_attention_packet(op, valid_tokens)
    return _bf16_random(shape)


def _profile_operator(
    *,
    name: str,
    op,
    warmup_iters: int,
    timed_iters: int,
    valid_tokens: int,
) -> tuple[float, float, float, float]:
    specs = op.get_arg_spec()
    buffer_names = [f"arg{idx}" for idx in range(len(specs))]
    input_args = [
        buffer_names[idx] for idx, spec in enumerate(specs) if spec.direction == "in"
    ]
    output_args = [
        buffer_names[idx] for idx, spec in enumerate(specs) if spec.direction == "out"
    ]
    if len(input_args) + len(output_args) != len(specs):
        raise ValueError(f"{name}: only in/out args are supported")

    fused_op = FusedMLIROperator(
        f"profile_{name}",
        [(op, *buffer_names)],
        input_args=input_args,
        output_args=output_args,
        compile_mode="full_elf_dynamic",
        context=op.context,
    ).compile()
    fused = fused_op.get_callable()

    for idx, spec in enumerate(specs):
        if spec.direction == "in":
            data = _input_for_op(
                op,
                idx,
                spec.shape,
                valid_tokens=valid_tokens,
            )
            fused.get_buffer(buffer_names[idx]).torch_view()[:] = data.flatten()
        elif spec.direction != "out":
            raise ValueError(f"{name}: unsupported arg direction {spec.direction}")

    for _ in range(warmup_iters):
        fused()

    samples_us: list[float] = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        fused()
        samples_us.append((time.perf_counter() - start) * 1e6)

    return (
        statistics.fmean(samples_us),
        statistics.median(samples_us),
        min(samples_us),
        max(samples_us),
    )


def _gemm(
    *,
    rows: int,
    k: int,
    n: int,
    context,
) -> GEMM:
    return GEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=8,
        tile_m=8,
        tile_k=64,
        tile_n=64,
        b_col_maj=False,
        separate_c_tiles=False,
        emulate_bf16_mmul_with_bfp16=False,
        context=context,
    )


def _operators(config: QwenShapeConfig, args: argparse.Namespace):
    rows = args.batch_rows
    context_root = Path(args.build_dir)
    x_elements = rows * config.emb_dim
    q_elements = rows * config.attn_dim
    kv_elements = rows * config.kv_dim
    ffn_elements = rows * config.hidden_dim

    def ctx(name: str) -> AIEContext:
        return AIEContext(build_dir=context_root / name)

    return [
        (
            "rms_norm_x",
            RMSNorm(
                size=x_elements,
                num_aie_columns=8,
                num_channels=1,
                tile_size=config.emb_dim,
                weighted=True,
                context=ctx("rms_norm_x"),
            ),
            2,
        ),
        (
            "gemm_q_proj",
            _gemm(rows=rows, k=config.emb_dim, n=config.attn_dim, context=ctx("gemm_q")),
            1,
        ),
        (
            "gemm_k_or_v_proj",
            _gemm(rows=rows, k=config.emb_dim, n=config.kv_dim, context=ctx("gemm_kv")),
            2,
        ),
        (
            "rms_norm_q",
            RMSNorm(
                size=q_elements,
                num_aie_columns=8,
                num_channels=1,
                tile_size=config.head_dim,
                weighted=True,
                context=ctx("rms_norm_q"),
            ),
            1,
        ),
        (
            "rms_norm_k",
            RMSNorm(
                size=kv_elements,
                num_aie_columns=8,
                num_channels=1,
                tile_size=config.head_dim,
                weighted=True,
                context=ctx("rms_norm_k"),
            ),
            1,
        ),
        (
            "rope_q",
            RoPE(
                rows=rows * config.n_heads,
                cols=config.head_dim,
                angle_rows=rows,
                context=ctx("rope_q"),
            ),
            1,
        ),
        (
            "rope_k",
            RoPE(
                rows=rows * config.n_kv_groups,
                cols=config.head_dim,
                angle_rows=rows,
                context=ctx("rope_k"),
            ),
            1,
        ),
        (
            "attention_one_lane",
            LlamaChunkedAttention(
                max_seq_len=args.max_seq_len,
                num_kv_groups=config.n_kv_groups,
                q_heads_per_group=config.q_heads_per_group,
                head_dim=config.head_dim,
                chunk_size=args.chunk_size,
                context=ctx("attention_one_lane"),
            ),
            args.batch_size,
        ),
        (
            "gemm_o_proj",
            _gemm(rows=rows, k=config.attn_dim, n=config.emb_dim, context=ctx("gemm_o")),
            1,
        ),
        (
            "residual_add",
            ElementwiseAdd(
                size=x_elements,
                tile_size=config.emb_dim // 8,
                num_aie_columns=8,
                context=ctx("residual_add"),
            ),
            2,
        ),
        (
            "gemm_ffn_gate_or_up",
            _gemm(
                rows=rows,
                k=config.emb_dim,
                n=config.hidden_dim,
                context=ctx("gemm_ffn_up"),
            ),
            2,
        ),
        (
            "silu_mul_ffn",
            SiLUMul(
                size=ffn_elements,
                tile_size=config.hidden_dim // 8,
                num_aie_columns=8,
                context=ctx("silu_mul_ffn"),
            ),
            1,
        ),
        (
            "gemm_ffn_down",
            _gemm(
                rows=rows,
                k=config.hidden_dim,
                n=config.emb_dim,
                context=ctx("gemm_ffn_down"),
            ),
            1,
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_model_dir = Path("/home/taowen/models/qwen3-0.6b")
    if not default_model_dir.exists():
        default_model_dir = Path(
            "/home/taowen/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-0.6B/snapshots/"
            "c1899de289a04d12100db370d81485cdf75e47ca"
        )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--valid-tokens", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=3)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_profile_qwen3_batch_decode_ops"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.model_dir)
    print("# Qwen3 Batch Decode Operator Profile")
    print()
    print("```text")
    print(f"model_dir: {args.model_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"batch_rows: {args.batch_rows}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"valid_tokens: {args.valid_tokens}")
    print(f"timing_source: host wall time around one-op full ELF runs")
    print("```")
    print()
    print("| operator | multiplicity/layer | mean ms | median ms | min ms | max ms | layer mean ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    layer_total_us = 0.0
    for name, op, multiplicity in _operators(config, args):
        mean_us, median_us, min_us, max_us = _profile_operator(
            name=name,
            op=op,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            valid_tokens=args.valid_tokens,
        )
        layer_mean_us = mean_us * multiplicity
        layer_total_us += layer_mean_us
        print(
            f"| {name} | {multiplicity} | "
            f"{mean_us / 1000.0:.3f} | "
            f"{median_us / 1000.0:.3f} | "
            f"{min_us / 1000.0:.3f} | "
            f"{max_us / 1000.0:.3f} | "
            f"{layer_mean_us / 1000.0:.3f} |"
        )

    print()
    print(f"estimated_layer_sum_ms: {layer_total_us / 1000.0:.3f}")
    print(f"estimated_28_layer_sum_ms: {layer_total_us * 28 / 1000.0:.3f}")


if __name__ == "__main__":
    main()
