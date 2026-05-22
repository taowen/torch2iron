#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MAX_SUPPORTED_SEQ_LEN = 2048
SEQ_LEN_BIN_SIZE = 512
MIN_COMPILED_SEQ_LEN = 512
DECODE_ATTN_CHUNK_SIZE = 64


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
