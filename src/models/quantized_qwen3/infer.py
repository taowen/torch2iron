#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run greedy reference inference on AutoRound W4A16 Qwen3."""

from __future__ import annotations

import argparse
import time

import torch

from models.quantized_qwen3.model import QuantizedQwenForCausalLM, find_model_dir


def _eos_ids(tokenizer, model) -> list[int]:
    eos = model.config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, list):
        return [int(item) for item in eos]
    return [int(eos)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference inference for W4A16 Qwen3")
    parser.add_argument("model_dir", help="Quantized model directory or parent output directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--compute-dtype",
        default="float32",
        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional torch CPU thread count for the reference runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    model_dir = find_model_dir(args.model_dir)
    t_load0 = time.perf_counter()
    model, tokenizer = QuantizedQwenForCausalLM.from_pretrained(
        model_dir,
        compute_dtype=args.compute_dtype,
    )
    t_load1 = time.perf_counter()

    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=_eos_ids(tokenizer, model),
        )
    t1 = time.perf_counter()

    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    new_tokens = max(0, output_ids.shape[1] - input_ids.shape[1])
    print(text)
    print("\n=== Quantized Qwen3 Reference Runtime ===")
    print(f"model_dir: {model_dir}")
    print(f"weight_source: {getattr(model, 'weight_source', 'unknown')}")
    print(f"load_time_s: {t_load1 - t_load0:.3f}")
    print(f"generate_time_s: {t1 - t0:.3f}")
    if new_tokens:
        print(f"tokens_per_second: {new_tokens / (t1 - t0):.3f}")
    print(f"new_tokens: {new_tokens}")


if __name__ == "__main__":
    main()
