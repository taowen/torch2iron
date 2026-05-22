#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import ml_dtypes
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.common.utils import XRTSubBuffer


class AIEPrefillBuffers:
    def __init__(self, prompt_len, emb_dim, hidden_dim, n_heads, n_kv_groups, head_dim):
        self.x = XRTTensor((prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.x_norm = XRTTensor((prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.attn_output = XRTTensor((prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_output = XRTTensor((prompt_len, emb_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_gate = XRTTensor((prompt_len, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_up = XRTTensor((prompt_len, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.ffn_hidden = XRTTensor((prompt_len, hidden_dim), dtype=ml_dtypes.bfloat16)
        self.queries = XRTTensor(
            (prompt_len * n_heads, head_dim), dtype=ml_dtypes.bfloat16
        )
        self.keys = XRTTensor(
            (prompt_len * n_kv_groups, head_dim), dtype=ml_dtypes.bfloat16
        )
        self.values = XRTTensor(
            (prompt_len, n_kv_groups * head_dim), dtype=ml_dtypes.bfloat16
        )
        self.rope_angles = XRTTensor((prompt_len, head_dim), dtype=ml_dtypes.bfloat16)
        self.attn_scores_queries_all = XRTTensor(
            (n_heads * prompt_len, head_dim), dtype=ml_dtypes.bfloat16
        )
        self.attn_scores_queries_per_head = [
            XRTSubBuffer.from_parent(
                self.attn_scores_queries_all,
                (prompt_len, head_dim),
                offset_elements=h * prompt_len * head_dim,
                length_elements=prompt_len * head_dim,
                dtype=ml_dtypes.bfloat16,
            )
            for h in range(n_heads)
        ]
        self.attn_scores_keys_all = XRTTensor(
            (n_kv_groups * head_dim, prompt_len), dtype=ml_dtypes.bfloat16
        )
        self.attn_scores_keys_per_kv_group = [
            XRTSubBuffer.from_parent(
                self.attn_scores_keys_all,
                (head_dim, prompt_len),
                offset_elements=g * head_dim * prompt_len,
                length_elements=head_dim * prompt_len,
                dtype=ml_dtypes.bfloat16,
            )
            for g in range(n_kv_groups)
        ]
        self.attn_scores = XRTTensor(
            (n_heads * prompt_len, prompt_len), dtype=ml_dtypes.bfloat16
        )
        self.attn_scores_per_head = [
            XRTSubBuffer.from_parent(
                self.attn_scores,
                (prompt_len, prompt_len),
                offset_elements=h * prompt_len * prompt_len,
                length_elements=prompt_len * prompt_len,
                dtype=ml_dtypes.bfloat16,
            )
            for h in range(n_heads)
        ]
        scale_factor = 1.0 / math.sqrt(head_dim)
        self.attn_scale_factor = XRTTensor(
            (n_heads * prompt_len, prompt_len), dtype=ml_dtypes.bfloat16
        )
        self.attn_scale_factor.fill_(scale_factor)
        self.attn_weights = XRTTensor(
            (n_heads * prompt_len, prompt_len), dtype=ml_dtypes.bfloat16
        )


class AIELlamaBuffers:
    def __init__(self, config, prompt_len, aie_ops):
        self.prefill = AIEPrefillBuffers(
            prompt_len,
            config.emb_dim,
            config.hidden_dim,
            config.n_heads,
            config.n_kv_groups,
            config.head_dim,
        )

        self.keys_cache = [
            XRTTensor(
                (config.n_kv_groups, prompt_len, config.head_dim),
                dtype=ml_dtypes.bfloat16,
            )
            for _ in range(config.n_layers)
        ]
        self.values_cache = [
            XRTTensor(
                (config.n_kv_groups, prompt_len, config.head_dim),
                dtype=ml_dtypes.bfloat16,
            )
            for _ in range(config.n_layers)
        ]

        self.W_norm1 = []
        self.W_norm2 = []
        self.W_attn_query_prefill = []
        self.W_attn_key_prefill = []
        self.W_attn_value_prefill = []
        self.W_ffn_gate_prefill = []
        self.W_ffn_up_prefill = []
        self.W_ffn_down_prefill = []
        for layer_idx in range(config.n_layers):
            self.W_norm1.append(
                XRTTensor.from_torch(
                    config.weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
                )
            )
            self.W_norm2.append(
                XRTTensor.from_torch(
                    config.weights[
                        f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                    ]
                )
            )
            self.W_attn_query_prefill.append(
                XRTTensor.from_torch(
                    config.weights[
                        f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                    ].T
                )
            )
            self.W_attn_key_prefill.append(
                XRTTensor.from_torch(
                    config.weights[
                        f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                    ].T
                )
            )
            self.W_attn_value_prefill.append(
                XRTTensor.from_torch(
                    config.weights[
                        f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                    ].T
                )
            )
            self.W_ffn_gate_prefill.append(
                XRTTensor.from_torch(
                    config.weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].T
                )
            )
            self.W_ffn_up_prefill.append(
                XRTTensor.from_torch(
                    config.weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"].T
                )
            )
            self.W_ffn_down_prefill.append(
                XRTTensor.from_torch(
                    config.weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"].T
                )
            )

        self.W_final_norm = XRTTensor.from_torch(config.weights["model.norm.weight"])
        self.W_out_head = XRTTensor.from_torch(
            config.weights["model.embed_tokens.weight"]
        )
        W_out_head_parts = aie_ops.prefill.gemv_out_head_compilable.partition_B(
            config.weights["model.embed_tokens.weight"]
            .view(torch.uint16)
            .numpy()
            .view(ml_dtypes.bfloat16),
            config.vocab_partitions,
        )
        self.W_out_head_parts = [
            XRTTensor(part, dtype=part.dtype) for part in W_out_head_parts
        ]
        self.prefill.logits = XRTTensor(
            (
                config.vocab_partitions,
                prompt_len,
                config.padded_vocab_size // config.vocab_partitions,
            ),
            dtype=ml_dtypes.bfloat16,
        )
        logits_part_len = prompt_len * (
            config.padded_vocab_size // config.vocab_partitions
        )
        self.prefill.logits_parts = [
            XRTSubBuffer.from_parent(
                self.prefill.logits,
                (
                    prompt_len,
                    config.padded_vocab_size // config.vocab_partitions,
                ),
                offset_elements=i * logits_part_len,
                length_elements=logits_part_len,
                dtype=ml_dtypes.bfloat16,
            )
            for i in range(config.vocab_partitions)
        ]
