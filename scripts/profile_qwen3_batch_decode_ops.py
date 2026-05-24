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
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from iron.common import MLIROperator
from iron.common.context import AIEContext
from torch2iron.fusion import FusedMLIROperator

from torch2iron.operators import (
    LlamaChunkedAttention,
    RMSNorm,
    RMSNormRoPE,
    ResidualAddRMSNorm,
    SiLUMul,
)

from models.quantized_qwen3.model import find_model_dir
from models.quantized_qwen3.operators.w4a16_gemm.op import (
    W4A16GEMM,
    W4A16KGroupGEMM,
    W4A16NShardGEMM,
    W4A16PairedKGroupGEMM,
)
from models.quantized_qwen3.packed_format import PackedInferenceStore, find_packed_dir


@dataclass(frozen=True)
class QwenShapeConfig:
    vocab_size: int
    emb_dim: int
    hidden_dim: int
    lm_head_gemm_out_features: int
    n_heads: int
    n_kv_groups: int
    head_dim: int
    group_size: int
    weight_store: PackedInferenceStore

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_groups * self.head_dim

    @property
    def q_heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_groups


@dataclass(frozen=True)
class OperatorProfile:
    name: str
    op: MLIROperator
    multiplicity: int
    fixed_inputs: dict[int, torch.Tensor]


@dataclass(frozen=True)
class OperatorTraffic:
    bf16_equivalent_weight_mb: float
    actual_weight_mb: float
    actual_total_mb: float


def _load_config(model_dir: Path) -> QwenShapeConfig:
    model_dir = find_model_dir(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    packed_dir = find_packed_dir(model_dir)
    if packed_dir is None:
        raise FileNotFoundError(
            f"missing qwen3_w4a16_packed artifact under {model_dir}; "
            "run `python -m models.quantized_qwen3.pack` first"
        )
    hf_config = json.loads(config_path.read_text())
    quant_config = hf_config.get("quantization_config") or {}
    weight_store = PackedInferenceStore(packed_dir)
    return QwenShapeConfig(
        vocab_size=int(hf_config["vocab_size"]),
        emb_dim=int(hf_config["hidden_size"]),
        hidden_dim=int(hf_config["intermediate_size"]),
        lm_head_gemm_out_features=weight_store.gemm_out_features("lm_head"),
        n_heads=int(hf_config["num_attention_heads"]),
        n_kv_groups=int(hf_config["num_key_value_heads"]),
        head_dim=int(
            hf_config.get(
                "head_dim",
                int(hf_config["hidden_size"]) // int(hf_config["num_attention_heads"]),
            )
        ),
        group_size=int(quant_config.get("group_size", 128)),
        weight_store=weight_store,
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
    fixed_inputs: dict[int, torch.Tensor],
) -> tuple[float, float, float, float]:
    specs = op.get_arg_spec()
    buffer_names = [f"arg{idx}" for idx in range(len(specs))]
    input_args = [
        buffer_names[idx]
        for idx, spec in enumerate(specs)
        if spec.direction == "in" and idx not in fixed_inputs
    ]
    output_args = [
        buffer_names[idx] for idx, spec in enumerate(specs) if spec.direction == "out"
    ]
    external_args = {
        f"fixed_input_{idx}": [buffer_names[idx]]
        for idx, spec in enumerate(specs)
        if spec.direction == "in" and idx in fixed_inputs
    }
    if len(input_args) + len(output_args) != len(specs):
        input_count = sum(1 for spec in specs if spec.direction == "in")
        if input_count + len(output_args) != len(specs):
            raise ValueError(f"{name}: only in/out args are supported")

    fused_op = FusedMLIROperator(
        f"profile_{name}",
        [(op, *buffer_names)],
        input_args=input_args,
        output_args=output_args,
        external_args=external_args,
        compile_mode="full_elf_dynamic",
        context=op.context,
    ).compile()
    fused = fused_op.get_callable()

    for idx, spec in enumerate(specs):
        if spec.direction == "in":
            if idx in fixed_inputs:
                data = fixed_inputs[idx].flatten()
                expected_elements = math.prod(int(dim) for dim in spec.shape)
                if data.numel() != expected_elements:
                    raise ValueError(
                        f"{name}: fixed input arg{idx} has {data.numel()} elements, "
                        f"expected {expected_elements}"
                    )
                arg_buffer = fused.get_buffer(buffer_names[idx])
                arg_buffer.torch_view()[:] = data
                arg_buffer.to("npu")
            else:
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


def _w4_gemm(
    config: QwenShapeConfig,
    *,
    rows: int,
    k: int,
    n: int,
    prefix: str,
    context,
) -> tuple[W4A16GEMM, dict[int, torch.Tensor]]:
    tile_m = 4 if rows <= 8 else 8
    op = W4A16GEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=8,
        num_aie_rows=rows // tile_m,
        tile_m=tile_m,
        tile_k=128,
        tile_n=64,
        group_size=config.group_size,
        context=context,
    )
    _linear_spec, w4_weight = config.weight_store.linear_gemm_w4_weight(prefix)
    return op, {1: w4_weight}


def _w4_k_group_gemm(
    config: QwenShapeConfig,
    *,
    rows: int,
    k: int,
    n: int,
    prefix: str,
    k_group: int = 2,
    context,
) -> tuple[W4A16KGroupGEMM, dict[int, torch.Tensor]]:
    tile_m = 4 if rows <= 8 else 8
    op = W4A16KGroupGEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=8,
        num_aie_rows=rows // tile_m,
        tile_m=tile_m,
        tile_k=128,
        tile_n=64,
        group_size=config.group_size,
        k_group=k_group,
        context=context,
    )
    _linear_spec, w4_weight = config.weight_store.linear_gemm_w4_weight(prefix)
    return op, {1: w4_weight}


def _w4_paired_k_group_gemm(
    config: QwenShapeConfig,
    *,
    rows: int,
    k: int,
    n: int,
    prefix: str,
    context,
) -> tuple[W4A16PairedKGroupGEMM, dict[int, torch.Tensor]]:
    tile_m = 4 if rows <= 8 else 8
    op = W4A16PairedKGroupGEMM(
        M=rows,
        K=k,
        N=n,
        num_aie_columns=8,
        num_aie_rows=rows // tile_m,
        tile_m=tile_m,
        tile_k=128,
        tile_n=64,
        group_size=config.group_size,
        k_group=2,
        context=context,
    )
    _linear_spec, w4_weight = config.weight_store.linear_paired_gemm_w4_weight(prefix)
    return op, {1: w4_weight}


def _n_shard_rows(n: int, tile_n: int) -> int:
    n_groups_per_col = n // (tile_n * 8)
    for rows in (4, 3, 2, 1):
        if n_groups_per_col % rows == 0:
            return rows
    raise ValueError(f"N={n} cannot be sharded across 8 columns")


def _w4_lm_head_gemm(
    config: QwenShapeConfig,
    *,
    rows: int,
    context,
) -> tuple[W4A16NShardGEMM, dict[int, torch.Tensor]]:
    tile_m = 4 if rows <= 8 else 8
    tile_n = 64
    op = W4A16NShardGEMM(
        M=rows,
        K=config.emb_dim,
        N=config.lm_head_gemm_out_features,
        num_aie_columns=8,
        num_aie_rows=_n_shard_rows(config.lm_head_gemm_out_features, tile_n),
        tile_m=tile_m,
        tile_k=128,
        tile_n=tile_n,
        group_size=config.group_size,
        context=context,
    )
    _linear_spec, w4_weight = config.weight_store.linear_gemm_w4_weight("lm_head")
    return op, {1: w4_weight}


def _operator_traffic(op) -> OperatorTraffic | None:
    if not isinstance(op, W4A16GEMM):
        return None

    pair_count = 2 if isinstance(op, W4A16PairedKGroupGEMM) else 1
    bf16_weight_bytes = pair_count * op.K * op.N * 2
    actual_weight_bytes = (
        pair_count * op.n_tiles * op.k_tiles * op.n_blocks * op.n_block_bytes
    )
    actual_total_bytes = actual_weight_bytes + op.M * op.K * 2 + pair_count * op.M * op.N * 2
    return OperatorTraffic(
        bf16_equivalent_weight_mb=bf16_weight_bytes / (1024 * 1024),
        actual_weight_mb=actual_weight_bytes / (1024 * 1024),
        actual_total_mb=actual_total_bytes / (1024 * 1024),
    )


def _operators(config: QwenShapeConfig, args: argparse.Namespace):
    rows = args.batch_rows
    context_root = (
        Path(args.build_dir)
        / f"rows{rows}_ctx{args.max_seq_len}_chunk{args.chunk_size}"
    )
    x_elements = rows * config.emb_dim
    q_elements = rows * config.attn_dim
    kv_elements = rows * config.kv_dim
    ffn_elements = rows * config.hidden_dim
    hidden_norm_columns = min(8, rows)
    rms_rope_columns = min(8, rows)

    def ctx(name: str) -> AIEContext:
        return AIEContext(build_dir=context_root / name)

    q_proj, q_proj_inputs = _w4_k_group_gemm(
        config,
        rows=rows,
        k=config.emb_dim,
        n=config.attn_dim,
        prefix="model.layers.0.self_attn.q_proj",
        k_group=4,
        context=ctx("w4_gemm_q"),
    )
    kv_proj, kv_proj_inputs = _w4_paired_k_group_gemm(
        config,
        rows=rows,
        k=config.emb_dim,
        n=config.kv_dim,
        prefix="model.layers.0.self_attn.kv_proj",
        context=ctx("w4_paired_gemm_kv"),
    )
    o_proj, o_proj_inputs = _w4_k_group_gemm(
        config,
        rows=rows,
        k=config.attn_dim,
        n=config.emb_dim,
        prefix="model.layers.0.self_attn.o_proj",
        context=ctx("w4_gemm_o"),
    )
    gate_up_proj, gate_up_proj_inputs = _w4_paired_k_group_gemm(
        config,
        rows=rows,
        k=config.emb_dim,
        n=config.hidden_dim,
        prefix="model.layers.0.mlp.gate_up_proj",
        context=ctx("w4_paired_gemm_ffn_up"),
    )
    down_proj, down_proj_inputs = _w4_k_group_gemm(
        config,
        rows=rows,
        k=config.hidden_dim,
        n=config.emb_dim,
        prefix="model.layers.0.mlp.down_proj",
        context=ctx("w4_gemm_ffn_down"),
    )
    lm_head, lm_head_inputs = _w4_lm_head_gemm(
        config,
        rows=rows,
        context=ctx("w4_gemm_lm_head"),
    )

    return [
        OperatorProfile(
            "initial_rms_norm_x",
            RMSNorm(
                size=x_elements,
                num_aie_columns=hidden_norm_columns,
                num_channels=1,
                tile_size=config.emb_dim,
                weighted=True,
                context=ctx("rms_norm_x"),
            ),
            0,
            {},
        ),
        OperatorProfile(
            "w4a16_gemm_q_proj",
            q_proj,
            1,
            q_proj_inputs,
        ),
        OperatorProfile(
            "w4a16_paired_k_group_gemm_kv_proj",
            kv_proj,
            1,
            kv_proj_inputs,
        ),
        OperatorProfile(
            "rms_norm_rope_q",
            RMSNormRoPE(
                rows=rows * config.n_heads,
                cols=config.head_dim,
                angle_rows=rows,
                num_aie_columns=rms_rope_columns,
                context=ctx("rms_norm_rope_q"),
            ),
            1,
            {},
        ),
        OperatorProfile(
            "rms_norm_rope_k",
            RMSNormRoPE(
                rows=rows * config.n_kv_groups,
                cols=config.head_dim,
                angle_rows=rows,
                num_aie_columns=rms_rope_columns,
                context=ctx("rms_norm_rope_k"),
            ),
            1,
            {},
        ),
        OperatorProfile(
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
            {},
        ),
        OperatorProfile(
            "w4a16_gemm_o_proj",
            o_proj,
            1,
            o_proj_inputs,
        ),
        OperatorProfile(
            "residual_add_rms_norm",
            ResidualAddRMSNorm(
                size=x_elements,
                tile_size=config.emb_dim,
                num_aie_columns=hidden_norm_columns,
                context=ctx("residual_add_rms_norm"),
            ),
            2,
            {},
        ),
        OperatorProfile(
            "w4a16_paired_k_group_gemm_ffn_gate_up",
            gate_up_proj,
            1,
            gate_up_proj_inputs,
        ),
        OperatorProfile(
            "silu_mul_ffn",
            SiLUMul(
                size=ffn_elements,
                tile_size=config.hidden_dim // 8,
                num_aie_columns=8,
                context=ctx("silu_mul_ffn"),
            ),
            1,
            {},
        ),
        OperatorProfile(
            "w4a16_gemm_ffn_down",
            down_proj,
            1,
            down_proj_inputs,
        ),
        OperatorProfile(
            "w4a16_gemm_lm_head",
            lm_head,
            1,
            lm_head_inputs,
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_model_dir = Path(
        "/home/taowen/models/qwen3-0.6b-w4a16-autogptq-script-smoke/"
        "c1899de289a04d12100db370d81485cdf75e47ca-w4g128"
    )
    if not default_model_dir.exists():
        default_model_dir = Path(
            "/home/taowen/models/qwen3-0.6b-w4a16-autogptq-smoke"
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
        "--operator",
        action="append",
        default=[],
        help="Profile only this operator name. Repeat to include multiple operators.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_profile_qwen3_batch_decode_ops"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.model_dir)
    selected_operators = set(args.operator)
    print("# Qwen3 Batch Decode Operator Profile")
    print()
    print("```text")
    print(f"model_dir: {args.model_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"batch_rows: {args.batch_rows}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"valid_tokens: {args.valid_tokens}")
    print("timing_source: host wall time around one-op full ELF runs")
    print("linear_operator: W4A16 fused dequant GEMM with real packed gemm_w4_weight")
    if selected_operators:
        print(f"selected_operators: {', '.join(sorted(selected_operators))}")
    print("```")
    print()
    print(
        "| operator | multiplicity/layer | mean ms | median ms | min ms | max ms | "
        "layer mean ms | bf16 equivalent weight MB | actual W4 tile MB | bf16 equivalent GB/s |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    layer_total_us = 0.0
    final_total_us = 0.0
    layer_weight_mb = 0.0
    layer_compressed_weight_mb = 0.0
    for profile in _operators(config, args):
        if selected_operators and profile.name not in selected_operators:
            continue
        mean_us, median_us, min_us, max_us = _profile_operator(
            name=profile.name,
            op=profile.op,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            valid_tokens=args.valid_tokens,
            fixed_inputs=profile.fixed_inputs,
        )
        layer_mean_us = mean_us * profile.multiplicity
        if profile.name == "w4a16_gemm_lm_head":
            final_total_us += layer_mean_us
        else:
            layer_total_us += layer_mean_us
        traffic = _operator_traffic(profile.op)
        if traffic is None:
            weight_mb = ""
            compressed_mb = ""
            weight_gb_s = ""
        else:
            layer_weight_mb += (
                traffic.bf16_equivalent_weight_mb * profile.multiplicity
            )
            layer_compressed_weight_mb += traffic.actual_weight_mb * profile.multiplicity
            weight_mb = f"{traffic.bf16_equivalent_weight_mb:.3f}"
            compressed_mb = f"{traffic.actual_weight_mb:.3f}"
            weight_gb_s = (
                f"{traffic.bf16_equivalent_weight_mb / (mean_us / 1e6) / 1024:.3f}"
            )
        print(
            f"| {profile.name} | {profile.multiplicity} | "
            f"{mean_us / 1000.0:.3f} | "
            f"{median_us / 1000.0:.3f} | "
            f"{min_us / 1000.0:.3f} | "
            f"{max_us / 1000.0:.3f} | "
            f"{layer_mean_us / 1000.0:.3f} | "
            f"{weight_mb} | "
            f"{compressed_mb} | "
            f"{weight_gb_s} |"
        )

    print()
    print(f"estimated_layer_sum_ms: {layer_total_us / 1000.0:.3f}")
    print(f"estimated_final_sum_ms: {final_total_us / 1000.0:.3f}")
    if layer_weight_mb:
        print(f"selected_linear_bf16_equivalent_weight_mb_per_layer: {layer_weight_mb:.3f}")
        print(
            "selected_linear_actual_w4_tile_weight_mb_per_layer: "
            f"{layer_compressed_weight_mb:.3f}"
        )
        print(
            "selected_linear_weight_compression_ratio: "
            f"{layer_weight_mb / layer_compressed_weight_mb:.3f}"
        )
    print(f"estimated_28_layer_sum_ms: {layer_total_us * 28 / 1000.0:.3f}")
    print(f"estimated_decode_sum_ms: {(layer_total_us * 28 + final_total_us) / 1000.0:.3f}")


if __name__ == "__main__":
    main()
