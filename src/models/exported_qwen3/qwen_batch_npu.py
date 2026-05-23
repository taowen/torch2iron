#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-batch Qwen3 NPU inference.

The current single-request runner keeps decode as a GEMV-heavy path.  This
runner keeps prefill simple and serial, then decodes a fixed batch in one ELF
with padded-row GEMMs so the matrix projections use more AIE work in parallel.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from models.exported_qwen3 import qwen_inference_harness as harness
from models.exported_qwen3.aie_operators import load_prefill_fused_weight_buffers
from models.exported_qwen3.batch_decode_fused import (
    BATCH_DECODE_ROWS,
    batch_packet_cache_names,
    build_batch_decode_fused_op,
    present_key_name,
    present_value_name,
)
from models.exported_qwen3.decode_packet_cache import (
    decode_packet_slot_offsets,
    sync_decode_packet_cache_slot,
)
from models.exported_qwen3.generated.decode_layout import (
    DECODE_PACKET_CACHE_NAMES,
    DECODE_TRANSFORMER_WEIGHT_NAMES,
    DECODE_WEIGHT_SPECS,
)
from models.exported_qwen3.generated.prefill_operators import build_prefill_fused_op
from models.exported_qwen3.prefill_runtime import prefill_forward_pass
from models.exported_qwen3.qwen_packed_weights import (
    default_qwen_packed_weights_dir,
    load_qwen_packed_segment,
    validate_qwen_packed_weight_artifact,
)
from models.exported_qwen3.runtime_config import (
    DECODE_ATTN_CHUNK_SIZE,
    select_compiled_seq_len,
    select_decode_context_len,
    select_prefill_chunk_config,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Fixed-batch Qwen3 NPU inference")
    parser.add_argument("weights_path", type=str)
    parser.add_argument("tokenizer_path", type=str)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text for one batch lane. Repeat exactly --batch-size times.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Text file with one prompt per line. Must contain exactly --batch-size prompts.",
    )
    parser.add_argument(
        "--packed-weights-dir",
        type=str,
        default=None,
        help="Optional packed Qwen weight artifact used as the source for weights.",
    )
    parser.add_argument(
        "--require-packed-weights",
        action="store_true",
        help="Require the packed weight artifact instead of reading safetensors.",
    )
    return parser.parse_args()


def _argmax_token(logits: torch.Tensor) -> int:
    return torch.argmax(logits[:, -1, :], dim=-1).item()


def _spec_by_name() -> dict[str, dict[str, object]]:
    return {spec["name"]: dict(spec) for spec in DECODE_WEIGHT_SPECS}


def _load_decode_weight_tensor(config, spec, manifest, packed_dir):
    if manifest is None:
        return config.weights[spec["source"]]

    segments = {segment["name"]: segment for segment in manifest["segments"]}
    segment = segments[spec["name"]]
    return load_qwen_packed_segment(packed_dir, manifest, spec["name"]).reshape(
        tuple(segment["shape"])
    )


def load_batch_decode_weight_buffers(config, fused):
    packed_dir = getattr(config, "packed_weights_dir", None)
    require_packed = getattr(config, "require_packed_weights", False)
    manifest = None
    if packed_dir is not None and Path(packed_dir).exists():
        manifest = validate_qwen_packed_weight_artifact(config, packed_dir)
        logging.info("Loading batch decode weights from packed artifact: %s", packed_dir)
    elif require_packed:
        raise FileNotFoundError(f"packed weights required but not found at {packed_dir}")
    else:
        logging.info("Loading batch decode weights from safetensors tensors")

    specs = _spec_by_name()
    for name in DECODE_TRANSFORMER_WEIGHT_NAMES:
        spec = specs[name]
        tensor = _load_decode_weight_tensor(config, spec, manifest, packed_dir)
        if tensor.ndim == 2:
            tensor = tensor.T.contiguous()
        fused.get_buffer(name).torch_view()[:] = tensor.flatten()


def _build_prefill_runtime(config, prefill_seq_len):
    prefill_config = select_prefill_chunk_config(prefill_seq_len)
    build_suffix = (
        f"seq{prefill_seq_len}"
        f"_chunk{prefill_config.chunk_size}"
        f"_rows{prefill_config.compute_rows}"
        f"_qhblk{prefill_config.q_head_block_size}"
    )
    fused_op = build_prefill_fused_op(
        config,
        prefill_seq_len,
        build_suffix,
        chunk_size=prefill_config.chunk_size,
        compute_rows=prefill_config.compute_rows,
        q_head_block_size=prefill_config.q_head_block_size,
    )
    fused = fused_op.get_callable()
    load_prefill_fused_weight_buffers(config, fused)
    fused.weight_buffer.to("npu")
    fused.scratch_buffer.to("npu")
    fused.output_buffer.to("npu")
    return SimpleNamespace(
        chunk_size=prefill_config.chunk_size,
        compute_rows=prefill_config.compute_rows,
        q_head_block_size=prefill_config.q_head_block_size,
        fused_op=fused_op,
        fused=fused,
    )


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
            for slot in range(valid_tokens):
                src_k_offset, src_v_offset, src_mask_offset = decode_packet_slot_offsets(
                    config,
                    prefill_max_seq_len,
                    group_idx,
                    slot,
                )
                dst_k_offset, dst_v_offset, dst_mask_offset = decode_packet_slot_offsets(
                    config,
                    batch_max_seq_len,
                    group_idx,
                    slot,
                )
                dst_packet[
                    dst_k_offset : dst_k_offset + config.head_dim
                ] = src_packet[src_k_offset : src_k_offset + config.head_dim]
                dst_packet[
                    dst_v_offset : dst_v_offset + config.head_dim
                ] = src_packet[src_v_offset : src_v_offset + config.head_dim]
                dst_packet[dst_mask_offset] = src_packet[src_mask_offset]

            _, _, current_mask_offset = decode_packet_slot_offsets(
                config,
                batch_max_seq_len,
                group_idx,
                batch_current_slot,
            )
            dst_packet[current_mask_offset] = 1.0

        dst_cache.to("npu")


class QwenBatchNpuRunner:
    def __init__(self, config, prefill_seq_len, decode_max_seq_len, batch_size):
        if batch_size <= 1:
            raise ValueError("batch inference requires batch_size > 1")
        if batch_size > BATCH_DECODE_ROWS:
            raise ValueError(
                f"batch_size {batch_size} exceeds padded rows {BATCH_DECODE_ROWS}"
            )

        self.config = config
        self.batch_size = batch_size
        self.prefill_max_seq_len = prefill_seq_len
        self.max_seq_len = decode_max_seq_len
        self.aie_ops = SimpleNamespace()
        self.aie_ops.prefill = _build_prefill_runtime(config, prefill_seq_len)

        batch_op, current_cache_slot = build_batch_decode_fused_op(
            config,
            decode_max_seq_len,
            batch_size,
            (
                f"batch{batch_size}_decode{decode_max_seq_len}"
                f"_chunk{DECODE_ATTN_CHUNK_SIZE}"
            ),
        )
        self.batch_fused_op = batch_op
        self.batch_fused = batch_op.get_callable()
        self.current_cache_slot = current_cache_slot
        load_batch_decode_weight_buffers(config, self.batch_fused)
        self.batch_fused.weight_buffer.to("npu")
        self.batch_fused.scratch_buffer.to("npu")
        self.batch_fused.output_buffer.to("npu")

        for name in batch_packet_cache_names(config, batch_size):
            packet_cache = self.batch_fused.get_buffer(name)
            packet_cache.torch_view().zero_()
            packet_cache.to("npu")

    def prefill_one(self, token_ids: list[int], batch_idx: int) -> tuple[int, int]:
        state = harness.QwenModelState(self.config)
        state.token_ids = torch.tensor([token_ids], dtype=torch.long)
        logits, _ = prefill_forward_pass(self, state)
        first_token = _argmax_token(logits)
        valid_tokens = len(token_ids)
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
        return first_token, valid_tokens

    def decode_step(self, tokens: list[int], num_preceding_tokens: list[int]):
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
        fused.input_buffer.to("cpu")

        x_input = fused.get_buffer("x").torch_view().view(BATCH_DECODE_ROWS, config.emb_dim)
        x_input.zero_()
        tok_emb_weight = config.weights["model.embed_tokens.weight"]
        token_tensor = torch.tensor(tokens, dtype=torch.long).view(1, self.batch_size)
        x = torch.nn.functional.embedding(token_tensor, tok_emb_weight)[0]
        x_input[: self.batch_size, :] = x

        rope_angles = fused.get_buffer("rope_angles").torch_view().view(
            BATCH_DECODE_ROWS,
            config.head_dim,
        )
        rope_angles.zero_()
        for batch_idx, position in enumerate(num_preceding_tokens):
            rope_angles[batch_idx, :] = config.angles[position]

        fused()

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
                packet_cache = fused.get_buffer(
                    f"packet_cache_{layer_idx}_{batch_idx}"
                )
                sync_decode_packet_cache_slot(
                    config,
                    self.max_seq_len,
                    packet_cache,
                    present_key,
                    present_value,
                    dst_slot,
                )

        hidden_out = fused.get_buffer("hidden_out").torch_view().view(
            BATCH_DECODE_ROWS,
            config.emb_dim,
        )
        lm_head = config.weights[config.lm_head_weight_source]
        logits = hidden_out[: self.batch_size, :].to(dtype=lm_head.dtype) @ lm_head.T
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
        path = Path(prompts_file)
        batch_prompts = [line for line in path.read_text().splitlines() if line]
    elif prompts is not None:
        batch_prompts = prompts
    else:
        prompt = harness.get_prompt(prompt_len)
        return [prompt for _ in range(batch_size)]

    if len(batch_prompts) != batch_size:
        raise ValueError(
            f"expected {batch_size} prompts, got {len(batch_prompts)}"
        )
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
    args = _parse_args()

    if args.num_tokens < 1:
        raise ValueError("--num-tokens must be at least 1")

    config = harness.QwenConfig(args.weights_path, args.tokenizer_path)
    packed_weights_dir = (
        Path(args.packed_weights_dir)
        if args.packed_weights_dir is not None
        else default_qwen_packed_weights_dir(args.weights_path)
    )
    config.packed_weights_dir = packed_weights_dir
    config.require_packed_weights = args.require_packed_weights

    seed = 1608560892
    torch.manual_seed(seed)

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
        "Using batch size %d, prefill sequence length %d, decode context %d "
        "for %d requested positions (%d max prompt tokens)",
        args.batch_size,
        prefill_seq_len,
        max_seq_len,
        required_seq_len,
        max_prompt_tokens,
    )

    runner = QwenBatchNpuRunner(config, prefill_seq_len, max_seq_len, args.batch_size)

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
        print(f"\n=== Batch {batch_idx} ===")
        print(prompt + text)

    prefill_s = t_prefill_stop - t_prefill_start
    decode_s = t_decode_stop - t_decode_start
    total_tokens = args.batch_size * args.num_tokens
    sys.stderr.write("\n=== Batch Performance Statistics ===\n")
    sys.stderr.write(f"[Prefill] Serial TTFT for batch: {prefill_s:7.3f} s\n")
    if args.num_tokens > 1:
        decode_tokens = args.batch_size * (args.num_tokens - 1)
        sys.stderr.write(
            f"[Decode]  Batch steps:           {args.num_tokens - 1:7d}\n"
        )
        sys.stderr.write(
            f"[Decode]  Time per batch step:  {decode_s / (args.num_tokens - 1):7.3f} s\n"
        )
        sys.stderr.write(
            f"[Decode]  Tokens per second:    {decode_tokens / decode_s:7.3f}\n"
        )
    total_s = prefill_s + decode_s
    sys.stderr.write(f"[Total]   Tokens per second:    {total_tokens / total_s:7.3f}\n")


if __name__ == "__main__":
    main()
