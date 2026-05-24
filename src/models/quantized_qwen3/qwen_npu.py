#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time

import torch

from models.quantized_qwen3 import qwen_inference_harness as harness
from models.quantized_qwen3.aie_operators import AIEQwenOperators
from models.quantized_qwen3.batch_decode_fused import (
    MAX_BATCH_DECODE_ROWS,
    present_key_name,
    present_value_name,
)
from models.quantized_qwen3.decode_packet_cache import (
    decode_packet_slot_offsets,
    sync_decode_packet_cache_slot,
)
from models.quantized_qwen3.generated.decode_layout import DECODE_PACKET_CACHE_NAMES
from models.quantized_qwen3.prefill_runtime import prefill_forward_pass
from models.quantized_qwen3.runtime_config import (
    DECODE_ATTN_CHUNK_SIZE,
    select_compiled_seq_len,
    select_decode_context_len,
)


def _argmax_token(logits: torch.Tensor) -> int:
    return torch.argmax(logits[:, -1, :], dim=-1).item()


def _copy_prefill_packet_cache_to_batch(
    config,
    *,
    prefill_fused,
    prefill_max_seq_len,
    batch_fused,
    batch_max_seq_len,
    batch_idx,
    valid_tokens,
    batch_current_slot,
):
    if valid_tokens > min(prefill_max_seq_len, batch_max_seq_len):
        raise ValueError(
            f"cannot copy {valid_tokens} KV tokens from seq{prefill_max_seq_len} "
            f"to batch seq{batch_max_seq_len}"
        )

    for layer_idx in range(config.n_layers):
        src_cache = prefill_fused.get_buffer(DECODE_PACKET_CACHE_NAMES[layer_idx])
        dst_cache = batch_fused.get_buffer(f"packet_cache_{layer_idx}_{batch_idx}")
        src_packet = src_cache.torch_view()
        dst_packet = dst_cache.torch_view()
        dst_packet.zero_()

        for group_idx in range(config.n_kv_groups):
            copied_tokens = 0
            while copied_tokens < valid_tokens:
                rows = min(DECODE_ATTN_CHUNK_SIZE, valid_tokens - copied_tokens)
                src_k_offset, src_v_offset, src_mask_offset = decode_packet_slot_offsets(
                    config,
                    prefill_max_seq_len,
                    group_idx,
                    copied_tokens,
                )
                dst_k_offset, dst_v_offset, dst_mask_offset = decode_packet_slot_offsets(
                    config,
                    batch_max_seq_len,
                    group_idx,
                    copied_tokens,
                )
                kv_elements = rows * config.head_dim
                dst_packet[
                    dst_k_offset : dst_k_offset + kv_elements
                ] = src_packet[src_k_offset : src_k_offset + kv_elements]
                dst_packet[
                    dst_v_offset : dst_v_offset + kv_elements
                ] = src_packet[src_v_offset : src_v_offset + kv_elements]
                dst_packet[
                    dst_mask_offset : dst_mask_offset + rows
                ] = src_packet[src_mask_offset : src_mask_offset + rows]
                copied_tokens += rows

            _, _, current_mask_offset = decode_packet_slot_offsets(
                config,
                batch_max_seq_len,
                group_idx,
                batch_current_slot,
            )
            dst_packet[current_mask_offset] = 1.0

        dst_cache.to("npu")


class QwenNpuRunner:
    def __init__(self, config, prefill_seq_len, decode_max_seq_len, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > MAX_BATCH_DECODE_ROWS:
            raise ValueError(
                f"batch_size {batch_size} exceeds padded rows {MAX_BATCH_DECODE_ROWS}"
            )

        self.config = config
        self.batch_size = batch_size
        self.prefill_max_seq_len = prefill_seq_len
        self.max_seq_len = decode_max_seq_len
        self.aie_ops = AIEQwenOperators(
            config,
            prefill_seq_len,
            decode_max_seq_len,
            batch_size,
        )
        self.batch_fused = self.aie_ops.decode.fused
        self.current_cache_slot = self.aie_ops.decode.current_cache_slot
        self.decode_rows = self.aie_ops.decode.decode_rows

    def forward_pass(self, config, state):
        if config is not self.config:
            raise ValueError("QwenNpuRunner was called with a different config")
        if self.batch_size != 1:
            raise ValueError("forward_pass is only defined for batch_size=1")

        _, seq_len = state.token_ids.shape
        if seq_len > 1:
            logits, state = self.prefill_state(state, batch_idx=0)
            state.num_preceding_tokens = seq_len
            return logits, state

        logits = self.decode_step(
            [int(state.token_ids[0, 0].item())],
            [state.num_preceding_tokens],
        )
        state.num_preceding_tokens += 1
        return logits, state

    def prefill_state(self, state, batch_idx: int) -> tuple[torch.Tensor, object]:
        logits, state = prefill_forward_pass(self, state)
        valid_tokens = state.token_ids.shape[1]
        _copy_prefill_packet_cache_to_batch(
            self.config,
            prefill_fused=self.aie_ops.prefill.fused,
            prefill_max_seq_len=self.prefill_max_seq_len,
            batch_fused=self.batch_fused,
            batch_max_seq_len=self.max_seq_len,
            batch_idx=batch_idx,
            valid_tokens=valid_tokens,
            batch_current_slot=self.current_cache_slot,
        )
        return logits, state

    def prefill_one(self, token_ids: list[int], batch_idx: int) -> tuple[int, int]:
        state = harness.QwenModelState(self.config)
        state.token_ids = torch.tensor([token_ids], dtype=torch.long)
        logits, _state = self.prefill_state(state, batch_idx)
        return _argmax_token(logits), len(token_ids)

    def decode_step(self, tokens: list[int], num_preceding_tokens: list[int]) -> torch.Tensor:
        if len(tokens) != self.batch_size:
            raise ValueError("token batch size does not match runner batch size")
        if len(num_preceding_tokens) != self.batch_size:
            raise ValueError("length batch size does not match runner batch size")
        if max(num_preceding_tokens) >= self.max_seq_len:
            raise ValueError(
                f"decode length {max(num_preceding_tokens)} exceeds seq{self.max_seq_len}"
            )

        fused = self.batch_fused
        config = self.config
        fused.mark_buffer_dirty("input")

        x_input = fused.get_buffer("x").torch_view().view(self.decode_rows, config.emb_dim)
        x_input.zero_()
        tok_emb_weight = config.weights["model.embed_tokens.weight"]
        token_tensor = torch.tensor(tokens, dtype=torch.long).view(1, self.batch_size)
        x = torch.nn.functional.embedding(token_tensor, tok_emb_weight)[0]
        x_input[: self.batch_size, :] = x

        rope_angles = fused.get_buffer("rope_angles").torch_view().view(
            self.decode_rows,
            config.head_dim,
        )
        rope_angles.zero_()
        for batch_idx, position in enumerate(num_preceding_tokens):
            rope_angles[batch_idx, :] = config.angles[position]

        profile_decode = os.environ.get("TORCH2IRON_PROFILE_DECODE") == "1"
        transformer_start = time.perf_counter() if profile_decode else 0.0
        fused()
        transformer_stop = time.perf_counter() if profile_decode else 0.0

        cache_start = time.perf_counter() if profile_decode else 0.0
        for batch_idx, dst_slot in enumerate(num_preceding_tokens):
            if dst_slot == self.current_cache_slot:
                continue
            for layer_idx in range(config.n_layers):
                present_key = (
                    fused.get_buffer(present_key_name(layer_idx, batch_idx))
                    .data
                    .reshape(config.n_kv_groups, config.head_dim)
                )
                present_value = (
                    fused.get_buffer(present_value_name(layer_idx, batch_idx))
                    .data
                    .reshape(config.n_kv_groups, config.head_dim)
                )
                packet_cache = fused.get_buffer(f"packet_cache_{layer_idx}_{batch_idx}")
                sync_decode_packet_cache_slot(
                    config,
                    self.max_seq_len,
                    packet_cache,
                    present_key,
                    present_value,
                    dst_slot,
                )
        cache_stop = time.perf_counter() if profile_decode else 0.0

        if profile_decode:
            logging.info(
                "decode profile: fused_transformer_lm_head=%.4fs cache=%.4fs",
                transformer_stop - transformer_start,
                cache_stop - cache_start,
            )
        logits = fused.get_buffer("logits").torch_view().view(
            self.decode_rows,
            config.lm_head_gemm_out_features,
        )[: self.batch_size, : config.vocab_size]
        return logits.view(self.batch_size, 1, config.vocab_size)


def _make_batch_prompts(
    *,
    prompt_len: int,
    batch_size: int,
    prompts: list[str] | None,
    prompts_file: str | None,
) -> list[str]:
    if prompts is not None and prompts_file is not None:
        raise ValueError("use either --prompt or --prompts-file, not both")

    if prompts_file is not None:
        batch_prompts = [line for line in Path(prompts_file).read_text().splitlines() if line]
    elif prompts is not None:
        batch_prompts = prompts
    else:
        prompt = harness.get_prompt(prompt_len)
        return [prompt for _ in range(batch_size)]

    if len(batch_prompts) != batch_size:
        raise ValueError(f"expected {batch_size} prompts, got {len(batch_prompts)}")
    return batch_prompts


def _init_batch(config, prompts: list[str]) -> list[list[int]]:
    token_batches = [
        config.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    for idx, token_ids in enumerate(token_batches):
        if len(token_ids) > config.context_length:
            raise ValueError(
                f"prompt {idx} has {len(token_ids)} tokens, exceeds "
                f"context length {config.context_length}"
            )
    return token_batches


def main():
    logging.basicConfig(level=logging.INFO)
    args = harness.parse_args()
    if args.num_tokens < 1:
        raise ValueError("--num-tokens must be at least 1")

    config = harness.QwenConfig(args.weights_path, args.tokenizer_path)
    prompts = _make_batch_prompts(
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
        prompts=args.prompt,
        prompts_file=args.prompts_file,
    )
    prompt_token_ids = _init_batch(config, prompts)
    max_prompt_tokens = max(len(tokens) for tokens in prompt_token_ids)
    required_seq_len = max_prompt_tokens + args.num_tokens
    prefill_seq_len = select_compiled_seq_len(max_prompt_tokens)
    max_seq_len = select_decode_context_len(required_seq_len)
    logging.info(
        "Using batch size %d, prefill sequence length %d and decode context %d "
        "for %d requested positions (%d max prompt tokens)",
        args.batch_size,
        prefill_seq_len,
        max_seq_len,
        required_seq_len,
        max_prompt_tokens,
    )

    runner = QwenNpuRunner(config, prefill_seq_len, max_seq_len, args.batch_size)

    generated: list[list[int]] = [[] for _ in range(args.batch_size)]
    num_preceding_tokens: list[int] = []
    first_tokens: list[int] = []

    t_prefill_start = time.perf_counter()
    for batch_idx, token_ids in enumerate(prompt_token_ids):
        first_token, valid_tokens = runner.prefill_one(token_ids, batch_idx)
        first_tokens.append(first_token)
        generated[batch_idx].append(first_token)
        num_preceding_tokens.append(valid_tokens)
    t_prefill_stop = time.perf_counter()

    next_tokens = first_tokens
    t_decode_start = time.perf_counter()
    for _ in range(args.num_tokens - 1):
        logits = runner.decode_step(next_tokens, num_preceding_tokens)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for batch_idx, token in enumerate(next_tokens):
            generated[batch_idx].append(int(token))
            num_preceding_tokens[batch_idx] += 1
    t_decode_stop = time.perf_counter()

    for batch_idx, prompt in enumerate(prompts):
        text = config.tokenizer.decode(
            generated[batch_idx],
            skip_special_tokens=False,
        )
        if args.batch_size == 1:
            print(prompt + text)
        else:
            print(f"\n=== Batch {batch_idx} ===")
            print(prompt + text)

    prefill_s = t_prefill_stop - t_prefill_start
    decode_s = t_decode_stop - t_decode_start
    total_tokens = args.batch_size * args.num_tokens
    sys.stderr.write("\n=== Performance Statistics ===\n")
    sys.stderr.write(f"[Prefill] Serial TTFT for batch: {prefill_s:7.3f} s\n")
    if args.num_tokens > 1:
        decode_tokens = args.batch_size * (args.num_tokens - 1)
        sys.stderr.write(
            f"[Decode]  Batch steps:          {args.num_tokens - 1:7d}\n"
        )
        sys.stderr.write(
            f"[Decode]  Time per batch step: {decode_s / (args.num_tokens - 1):7.3f} s\n"
        )
        sys.stderr.write(
            f"[Decode]  Tokens per second:   {decode_tokens / decode_s:7.3f}\n"
        )
    total_s = prefill_s + decode_s
    sys.stderr.write(f"[Total]   Tokens per second:   {total_tokens / total_s:7.3f}\n")


if __name__ == "__main__":
    main()
