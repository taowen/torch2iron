#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline W4A16 quantization for Qwen3 using vendored AutoRound."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from models.quantized_qwen3.packed_format import write_packed_inference_artifact


DEFAULT_CALIBRATION_TEXTS = [
    (
        "The quick brown fox jumps over the lazy dog. This calibration paragraph "
        "contains common English words, punctuation, and repeated clauses so a "
        "short smoke-test sequence has enough tokens for quantization. AutoRound "
        "uses these tokens to run a small forward pass and calibrate rounding. "
        "The quick brown fox jumps over the lazy dog. This calibration paragraph "
        "is repeated so it survives sequence-length filtering during a smoke test."
    ),
    (
        "请解释本地大语言模型推理、权重量化、KV cache、prefill 和 decode 的区别。"
        "这段中文校准文本用于快速验证离线量化流程，而不是用于正式精度评估。"
        "为了确保分词后的长度足够，这里继续重复一些常见词：本地推理、注意力、"
        "权重加载、离线量化、校准文本、矩阵乘法、解码阶段、缓存更新、性能验证。"
    ),
    (
        "AutoRound learns rounding decisions and clipping ranges from calibration "
        "text. A tiny calibration set is enough to test export plumbing, but a "
        "production model should use more samples and longer sequences. The smoke "
        "test only proves that qweight, qzeros, and scales can be exported in the "
        "AutoGPTQ layout and later consumed by a reference runtime."
    ),
    (
        "Write a Python function that computes attention from query, key, and "
        "value tensors, then discuss tiled attention and fused dequantized GEMM. "
        "The implementation should mention batch decode, group quantization, "
        "signed int4 weights, bf16 activations, scale lookup, and memory bandwidth."
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_default_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CALIBRATION_TEXTS, ensure_ascii=False) + "\n")
    return path


def _submodule_pythonpath(env: dict[str, str]) -> dict[str, str]:
    repo_root = _repo_root()
    auto_round_path = repo_root / "third_party" / "auto-round"
    pythonpath = str(auto_round_path)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def find_quantized_model_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    if (output_dir / "model.safetensors").exists():
        return output_dir
    candidates = sorted(
        [path for path in output_dir.rglob("model.safetensors")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no model.safetensors found under {output_dir}")
    return candidates[0].parent


def run_autoround_quantization(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else None
    if dataset is None:
        dataset = _write_default_dataset(output_dir / "calibration_smoke.json")

    cmd = [
        sys.executable,
        "-m",
        "auto_round",
        str(Path(args.model).expanduser()),
        "--scheme",
        "W4A16",
        "--format",
        "auto_gptq",
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--iters",
        str(args.iters),
        "--nsamples",
        str(args.nsamples),
        "--seqlen",
        str(args.seqlen),
        "--batch_size",
        str(args.batch_size),
        "--group_size",
        "128",
        "--dataset",
        str(dataset),
    ]
    if args.platform:
        cmd.extend(["--platform", args.platform])
    if args.quant_lm_head:
        cmd.append("--quant_lm_head")
    if args.disable_opt_rtn:
        cmd.append("--disable_opt_rtn")

    env = _submodule_pythonpath(os.environ.copy())
    if args.modelscope:
        env["AR_USE_MODELSCOPE"] = "1"

    subprocess.run(cmd, check=True, env=env)
    model_dir = find_quantized_model_dir(output_dir)
    manifest = write_packed_inference_artifact(model_dir)
    packed_dir = model_dir / "qwen3_w4a16_packed"
    print(f"packed_inference_dir: {packed_dir}")
    print(f"packed_inference_bytes: {manifest['total_bytes']}")
    return model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Qwen3 and write the packed W4A16 inference format")
    parser.add_argument("model", help="HF/ModelScope model id or local model directory")
    parser.add_argument("--output-dir", required=True, help="Directory for AutoRound output")
    parser.add_argument("--dataset", default=None, help="Local json/jsonl calibration file")
    parser.add_argument("--device", default="cpu", help="AutoRound device, e.g. cpu, cuda:0, xpu:0")
    parser.add_argument("--platform", default=None, choices=["hf", "model_scope"])
    parser.add_argument("--modelscope", action="store_true", help="Set AR_USE_MODELSCOPE=1")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quant-lm-head", action="store_true")
    parser.add_argument(
        "--disable-opt-rtn",
        action="store_true",
        help="Disable AutoRound optimized RTN when running with --iters 0",
    )
    return parser.parse_args()


def main() -> None:
    model_dir = run_autoround_quantization(parse_args())
    print(f"quantized_model_dir: {model_dir}")


if __name__ == "__main__":
    main()
