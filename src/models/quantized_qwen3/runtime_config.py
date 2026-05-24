#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

MAX_SUPPORTED_SEQ_LEN = 2048
SEQ_LEN_BIN_SIZE = 64
MIN_COMPILED_SEQ_LEN = 64
DECODE_ATTN_CHUNK_SIZE = 64
DECODE_VARIANT_SEQ_LENS = (64, 128, 256, 512, 1024, 2048)
PREFILL_CHUNK_COMPUTE_ROWS = 32
SHORT_PREFILL_CHUNK_SIZE = 8
SHORT_PREFILL_Q_HEAD_BLOCK_SIZE = 2
MEDIUM_PREFILL_CHUNK_SIZE = 16
MEDIUM_PREFILL_Q_HEAD_BLOCK_SIZE = 1
LONG_PREFILL_CHUNK_SIZE = 16
LONG_PREFILL_Q_HEAD_BLOCK_SIZE = 1
LONG_PREFILL_MIN_COMPILED_SEQ_LEN = 256


@dataclass(frozen=True)
class PrefillChunkConfig:
    chunk_size: int
    compute_rows: int
    q_head_block_size: int


def select_compiled_seq_len(required_tokens):
    if required_tokens > MAX_SUPPORTED_SEQ_LEN:
        raise ValueError(
            f"required sequence length {required_tokens} exceeds "
            f"MAX_SUPPORTED_SEQ_LEN={MAX_SUPPORTED_SEQ_LEN}"
        )
    rounded = (
        (required_tokens + SEQ_LEN_BIN_SIZE - 1)
        // SEQ_LEN_BIN_SIZE
        * SEQ_LEN_BIN_SIZE
    )
    return max(MIN_COMPILED_SEQ_LEN, rounded)


def _validate_static_seq_len(seq_len):
    if seq_len % DECODE_ATTN_CHUNK_SIZE != 0:
        raise ValueError(
            f"static sequence length {seq_len} must be a multiple of "
            f"DECODE_ATTN_CHUNK_SIZE={DECODE_ATTN_CHUNK_SIZE}"
        )


def iter_decode_variant_seq_lens(max_seq_len, extra_seq_lens=()):
    _validate_static_seq_len(max_seq_len)
    variants = {
        seq_len
        for seq_len in DECODE_VARIANT_SEQ_LENS
        if MIN_COMPILED_SEQ_LEN <= seq_len <= max_seq_len
    }
    variants.add(max_seq_len)
    for seq_len in extra_seq_lens:
        if seq_len is None:
            continue
        _validate_static_seq_len(seq_len)
        if MIN_COMPILED_SEQ_LEN <= seq_len <= max_seq_len:
            variants.add(seq_len)
    return tuple(sorted(variants))


def select_decode_context_len(required_tokens):
    if required_tokens > MAX_SUPPORTED_SEQ_LEN:
        raise ValueError(
            f"required sequence length {required_tokens} exceeds "
            f"MAX_SUPPORTED_SEQ_LEN={MAX_SUPPORTED_SEQ_LEN}"
        )
    for seq_len in DECODE_VARIANT_SEQ_LENS:
        if required_tokens <= seq_len:
            return seq_len
    raise ValueError(f"no decode context can cover {required_tokens} tokens")


def select_decode_variant_seq_len(required_tokens, max_seq_len, variant_seq_lens=None):
    if required_tokens > max_seq_len:
        raise ValueError(
            f"required sequence length {required_tokens} exceeds "
            f"compiled decode context {max_seq_len}"
        )
    variants = (
        iter_decode_variant_seq_lens(max_seq_len)
        if variant_seq_lens is None
        else tuple(sorted(variant_seq_lens))
    )
    for seq_len in variants:
        if required_tokens <= seq_len:
            return seq_len
    raise ValueError(f"no decode variant can cover {required_tokens} tokens")


def select_prefill_chunk_config(compiled_seq_len):
    if compiled_seq_len >= LONG_PREFILL_MIN_COMPILED_SEQ_LEN:
        return PrefillChunkConfig(
            chunk_size=LONG_PREFILL_CHUNK_SIZE,
            compute_rows=PREFILL_CHUNK_COMPUTE_ROWS,
            q_head_block_size=LONG_PREFILL_Q_HEAD_BLOCK_SIZE,
        )
    if compiled_seq_len > MIN_COMPILED_SEQ_LEN:
        return PrefillChunkConfig(
            chunk_size=MEDIUM_PREFILL_CHUNK_SIZE,
            compute_rows=PREFILL_CHUNK_COMPUTE_ROWS,
            q_head_block_size=MEDIUM_PREFILL_Q_HEAD_BLOCK_SIZE,
        )
    return PrefillChunkConfig(
        chunk_size=SHORT_PREFILL_CHUNK_SIZE,
        compute_rows=PREFILL_CHUNK_COMPUTE_ROWS,
        q_head_block_size=SHORT_PREFILL_Q_HEAD_BLOCK_SIZE,
    )
