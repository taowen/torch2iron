#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference harness for Qwen3 0.6B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import safetensors.torch
import torch
from transformers import AutoTokenizer


class QwenConfig:
    def __init__(self, weights_path, tokenizer_path):
        weights_path = Path(weights_path)
        tokenizer_path = Path(tokenizer_path)
        model_dir = weights_path if weights_path.is_dir() else weights_path.parent
        tokenizer_dir = tokenizer_path if tokenizer_path.is_dir() else tokenizer_path.parent
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"missing Qwen config: {config_path}")

        hf_config = json.loads(config_path.read_text())

        self.vocab_size = int(hf_config["vocab_size"])
        self.emb_dim = int(hf_config["hidden_size"])
        self.n_layers = int(hf_config["num_hidden_layers"])
        self.n_heads = int(hf_config["num_attention_heads"])
        self.n_kv_groups = int(hf_config["num_key_value_heads"])
        self.head_dim = int(hf_config.get("head_dim", self.emb_dim // self.n_heads))
        self.hidden_dim = int(hf_config["intermediate_size"])
        self.rms_norm_eps = float(hf_config.get("rms_norm_eps", 1e-6))

        self.rope_base = float(hf_config.get("rope_theta", 1000000.0))
        self.context_length = int(hf_config.get("max_position_embeddings", 40960))

        self.temperature = 0.0
        self.top_k = 1
        self.bos_token_id = hf_config.get("bos_token_id")
        self.eos_token_id = hf_config.get("eos_token_id")

        if weights_path.is_dir():
            weights_path = weights_path / "model.safetensors"
        self.weights = safetensors.torch.load_file(weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
        )
        self.lm_head_weight_source = (
            "lm_head.weight"
            if "lm_head.weight" in self.weights
            else "model.embed_tokens.weight"
        )
        self.angles = compute_rope_angles(
            self.head_dim,
            self.context_length,
            self.rope_base,
        )


class QwenModelState:
    def __init__(self, config):
        self.token_ids = torch.empty(0, dtype=torch.long)
        self.reset_kv_cache()

    def reset_kv_cache(self):
        self.num_preceding_tokens = 0


def compute_rope_angles(head_dim, context_length, rope_base=1000000.0):
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    position = torch.arange(context_length).float()
    freqs = torch.outer(position, inv_freq)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    angles = torch.empty(context_length, head_dim)
    angles[:, ::2] = cos
    angles[:, 1::2] = sin
    return angles


def generate_token(config, forward_pass, state):
    logits, state = forward_pass(config, state)
    last_token_logits = logits[:, -1, :]

    if config.temperature <= 0 or config.top_k == 1:
        return torch.argmax(last_token_logits, dim=-1).item(), state

    if config.temperature > 0:
        last_token_logits = last_token_logits / config.temperature

    if config.top_k is not None:
        top_logits, _ = torch.topk(last_token_logits, config.top_k)
        min_val = top_logits[:, -1:]
        last_token_logits = torch.where(
            last_token_logits < min_val,
            torch.tensor(float("-inf")),
            last_token_logits,
        )

    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token.item(), state


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 0.6B Inference Harness")
    parser.add_argument(
        "weights_path",
        type=str,
        help="Path to model.safetensors or a model directory.",
    )
    parser.add_argument(
        "tokenizer_path",
        type=str,
        help="Path to tokenizer directory or tokenizer.json.",
    )
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--num-tokens", type=int, default=40)
    parser.add_argument(
        "--prepare-weights",
        action="store_true",
        help="Write the packed Qwen bf16 weight artifact and exit.",
    )
    parser.add_argument(
        "--packed-weights-dir",
        type=str,
        default=None,
        help="Directory containing or receiving qwen_iron_packed weights.",
    )
    parser.add_argument(
        "--require-packed-weights",
        action="store_true",
        help="Require decode to load the packed weight artifact.",
    )
    return parser.parse_args()


def get_prompt(prompt_len):
    with open(Path(__file__).parent / "prompt.txt", "r") as f:
        prompt = f.read()
    return prompt[:prompt_len]


def init(
    weights_path,
    tokenizer_path,
    prompt="The capital of France is ",
):
    config = QwenConfig(weights_path, tokenizer_path)
    state = QwenModelState(config)

    seed = 1608560892
    torch.manual_seed(seed)

    prompt_token_ids = config.tokenizer.encode(prompt, add_special_tokens=False)
    assert (
        len(prompt_token_ids) <= config.context_length
    ), f"Prompt length ({len(prompt_token_ids)} tokens) exceeds model context length ({config.context_length})"
    state.token_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
    return config, state


def generate(config, state, forward_pass, num_tokens=100, use_kv_cache=True):
    n_tokens_generated = 0
    t_prefill_start = time.perf_counter()
    first_token, state = generate_token(config, forward_pass, state)
    token_text = config.tokenizer.decode([first_token], skip_special_tokens=False)
    n_tokens_generated += 1
    print(token_text, end="", flush=True)
    t_prefill_stop = time.perf_counter()

    if use_kv_cache:
        state.token_ids = torch.tensor([[first_token]], dtype=torch.long)
    else:
        state.reset_kv_cache()
        state.token_ids = torch.cat(
            [state.token_ids, torch.tensor([[first_token]], dtype=torch.long)],
            dim=1,
        )

    t_decode_start = time.perf_counter()
    for _ in range(num_tokens - 1):
        next_token, state = generate_token(config, forward_pass, state)
        token_text = config.tokenizer.decode([next_token], skip_special_tokens=False)
        n_tokens_generated += 1
        print(token_text, end="", flush=True)
        if use_kv_cache:
            state.token_ids = torch.tensor([[next_token]], dtype=torch.long)
        else:
            state.reset_kv_cache()
            state.token_ids = torch.cat(
                [state.token_ids, torch.tensor([[next_token]], dtype=torch.long)],
                dim=1,
            )
    t_decode_end = time.perf_counter()

    t_prefill = t_prefill_stop - t_prefill_start
    t_decode = t_decode_end - t_decode_start
    sys.stderr.write("\n\n=== Performance Statistics ===\n")
    sys.stderr.write(f"[Prefill] Time to first token:   {t_prefill:7.3f} s\n")
    if n_tokens_generated > 1:
        sys.stderr.write(
            f"[Decode]  Time per token (mean): {t_decode / (n_tokens_generated - 1):7.3f} s\n"
        )
        sys.stderr.write(
            f"[Decode]  Tokens per second:     {(n_tokens_generated - 1) / t_decode:7.3f}\n"
        )
    sys.stderr.write(
        f"[Total]   Time per token (mean): {(t_prefill + t_decode) / n_tokens_generated:7.3f} s\n"
    )
    sys.stderr.write(
        f"[Total]   Tokens per second:     {n_tokens_generated / (t_prefill + t_decode):7.3f}\n"
    )
