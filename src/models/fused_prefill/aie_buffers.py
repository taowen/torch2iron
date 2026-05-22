#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import ml_dtypes
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.common.utils import XRTSubBuffer

from models.fused_prefill.generated.prefill_layout import (
    EXPECTED_PREFILL_LAYERS,
    PREFILL_LAYER_WEIGHT_SPECS,
)

BF16 = ml_dtypes.bfloat16


def _bf16_tensor(shape):
    return XRTTensor(shape, dtype=BF16)


def _bf16_subbuffer(parent, shape, offset_elements, length_elements):
    return XRTSubBuffer.from_parent(
        parent,
        shape,
        offset_elements=offset_elements,
        length_elements=length_elements,
        dtype=BF16,
    )


def _allocate_named_tensors(target, shape_by_name):
    for name, shape in shape_by_name.items():
        setattr(target, name, _bf16_tensor(shape))


def _prefill_activation_shapes(config, prompt_len):
    return {
        "x": (prompt_len, config.emb_dim),
        "x_norm": (prompt_len, config.emb_dim),
        "attn_output": (prompt_len, config.emb_dim),
        "ffn_output": (prompt_len, config.emb_dim),
        "ffn_gate": (prompt_len, config.hidden_dim),
        "ffn_up": (prompt_len, config.hidden_dim),
        "ffn_hidden": (prompt_len, config.hidden_dim),
        "queries": (prompt_len * config.n_heads, config.head_dim),
        "keys": (prompt_len * config.n_kv_groups, config.head_dim),
        "values": (prompt_len, config.n_kv_groups * config.head_dim),
        "rope_angles": (prompt_len, config.head_dim),
        "attn_scores_queries_all": (config.n_heads * prompt_len, config.head_dim),
        "attn_scores_keys_all": (
            config.n_kv_groups * config.head_dim,
            prompt_len,
        ),
        "attn_scores": (config.n_heads * prompt_len, prompt_len),
        "attn_scale_factor": (config.n_heads * prompt_len, prompt_len),
        "attn_weights": (config.n_heads * prompt_len, prompt_len),
    }


def _contiguous_subbuffers(parent, count, shape, elements_per_item):
    return [
        _bf16_subbuffer(
            parent,
            shape,
            offset_elements=i * elements_per_item,
            length_elements=elements_per_item,
        )
        for i in range(count)
    ]


def _allocate_contiguous_subbuffers(target, specs):
    for attr_name, parent, count, shape, elements_per_item in specs:
        setattr(
            target,
            attr_name,
            _contiguous_subbuffers(parent, count, shape, elements_per_item),
        )


def _prefill_subbuffer_specs(prefill, config, prompt_len):
    return (
        (
            "attn_scores_queries_per_head",
            prefill.attn_scores_queries_all,
            config.n_heads,
            (prompt_len, config.head_dim),
            prompt_len * config.head_dim,
        ),
        (
            "attn_scores_keys_per_kv_group",
            prefill.attn_scores_keys_all,
            config.n_kv_groups,
            (config.head_dim, prompt_len),
            config.head_dim * prompt_len,
        ),
        (
            "attn_scores_per_head",
            prefill.attn_scores,
            config.n_heads,
            (prompt_len, prompt_len),
            prompt_len * prompt_len,
        ),
    )


def _layer_weight(config, layer_idx, suffix, transpose=False):
    tensor = config.weights[f"model.layers.{layer_idx}.{suffix}"]
    if transpose:
        tensor = tensor.T
    return XRTTensor.from_torch(tensor)


def _layer_weights(config, suffix, transpose=False):
    return [
        _layer_weight(config, layer_idx, suffix, transpose=transpose)
        for layer_idx in range(config.n_layers)
    ]


def _cache_buffers(config, prompt_len):
    return [
        _bf16_tensor((config.n_kv_groups, prompt_len, config.head_dim))
        for _ in range(config.n_layers)
    ]


def _assign_prefill_layer_weights(target, config):
    if config.n_layers != EXPECTED_PREFILL_LAYERS:
        raise ValueError(
            f"generated prefill buffer layout expects {EXPECTED_PREFILL_LAYERS} "
            f"layers, got {config.n_layers}"
        )
    for attr_name, suffix, transpose in PREFILL_LAYER_WEIGHT_SPECS:
        setattr(target, attr_name, _layer_weights(config, suffix, transpose=transpose))


def _embedding_weight(config):
    return config.weights["model.embed_tokens.weight"]


def _assign_global_weights(target, config):
    target.W_final_norm = XRTTensor.from_torch(config.weights["model.norm.weight"])
    target.W_out_head = XRTTensor.from_torch(_embedding_weight(config))


def _partition_lm_head(aie_ops, config):
    return aie_ops.prefill.gemv_out_head_compilable.partition_B(
        _embedding_weight(config).view(torch.uint16).numpy().view(BF16),
        config.vocab_partitions,
    )


def _assign_lm_head_buffers(target, config, prompt_len, aie_ops):
    target.W_out_head_parts = [
        XRTTensor(part, dtype=part.dtype) for part in _partition_lm_head(aie_ops, config)
    ]

    partition_width = config.padded_vocab_size // config.vocab_partitions
    target.prefill.logits = _bf16_tensor(
        (config.vocab_partitions, prompt_len, partition_width)
    )
    target.prefill.logits_parts = _contiguous_subbuffers(
        target.prefill.logits,
        config.vocab_partitions,
        (prompt_len, partition_width),
        prompt_len * partition_width,
    )


class AIEPrefillBuffers:
    def __init__(self, config, prompt_len):
        _allocate_named_tensors(self, _prefill_activation_shapes(config, prompt_len))
        _allocate_contiguous_subbuffers(
            self, _prefill_subbuffer_specs(self, config, prompt_len)
        )
        self.attn_scale_factor.fill_(1.0 / math.sqrt(config.head_dim))


class AIELlamaBuffers:
    def __init__(self, config, prompt_len, aie_ops):
        self.prefill = AIEPrefillBuffers(config, prompt_len)

        self.keys_cache = _cache_buffers(config, prompt_len)
        self.values_cache = _cache_buffers(config, prompt_len)

        _assign_prefill_layer_weights(self, config)
        _assign_global_weights(self, config)
        _assign_lm_head_buffers(self, config, prompt_len, aie_ops)
