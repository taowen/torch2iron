#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated prefill layout constants for quantized_qwen3.

Regenerate with:
    uv run python -m torch2iron.export.codegen --model-package models.quantized_qwen3

This file is rendered directly from torch.export.ExportedProgram and intentionally
does not import IRON runtime modules.
"""

from __future__ import annotations


EXPECTED_PREFILL_LAYERS = 28

PREFILL_LAYER_WEIGHT_SPECS = (
    (
        "W_norm1",
        "input_layernorm.weight",
        False,
    ),
    (
        "W_attn_query_prefill",
        "self_attn.q_proj.weight",
        True,
    ),
    (
        "W_attn_key_prefill",
        "self_attn.k_proj.weight",
        True,
    ),
    (
        "W_attn_value_prefill",
        "self_attn.v_proj.weight",
        True,
    ),
    (
        "W_attn_query_norm",
        "self_attn.q_norm.weight",
        False,
    ),
    (
        "W_attn_key_norm",
        "self_attn.k_norm.weight",
        False,
    ),
    (
        "W_attn_output_prefill",
        "self_attn.o_proj.weight",
        True,
    ),
    (
        "W_norm2",
        "post_attention_layernorm.weight",
        False,
    ),
    (
        "W_ffn_gate_prefill",
        "mlp.gate_proj.weight",
        True,
    ),
    (
        "W_ffn_up_prefill",
        "mlp.up_proj.weight",
        True,
    ),
    (
        "W_ffn_down_prefill",
        "mlp.down_proj.weight",
        True,
    ),
)
