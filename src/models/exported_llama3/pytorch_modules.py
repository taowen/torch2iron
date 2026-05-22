#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small PyTorch Llama modules used only for ``torch.export`` inspection.

These modules are not wired into the runtime path.  They keep the Llama block
structure and a few important semantic boundaries visible so
``dump_exported_program.py`` can show what PyTorch export actually emits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.library.custom_op("torch2iron::swiglu", mutates_args=())
def _swiglu_impl(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


@_swiglu_impl.register_fake
def _(gate: Tensor, up: Tensor) -> Tensor:
    del up
    return gate.new_empty(gate.shape)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return torch.ops.torch2iron.swiglu.default(gate, up)


def _rope_reference(x: Tensor, angles: Tensor) -> Tensor:
    cos = angles[..., ::2].repeat_interleave(2, dim=-1)
    sin = angles[..., 1::2].repeat_interleave(2, dim=-1)

    leading = [1] * (x.dim() - 3)
    cos = cos.view(*leading, angles.shape[-2], 1, angles.shape[-1])
    sin = sin.view(*leading, angles.shape[-2], 1, angles.shape[-1])

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rotated * sin


@torch.library.custom_op("torch2iron::rope", mutates_args=())
def _rope_impl(x: Tensor, angles: Tensor) -> Tensor:
    return _rope_reference(x, angles)


@_rope_impl.register_fake
def _(x: Tensor, angles: Tensor) -> Tensor:
    del angles
    return x.new_empty(x.shape)


def rope(x: Tensor, angles: Tensor) -> Tensor:
    return torch.ops.torch2iron.rope.default(x, angles)


@torch.library.custom_op("torch2iron::gqa_repeat", mutates_args=())
def _gqa_repeat_impl(x: Tensor, repeats: int) -> Tensor:
    return x.repeat_interleave(repeats, dim=1)


@_gqa_repeat_impl.register_fake
def _(x: Tensor, repeats: int) -> Tensor:
    shape = list(x.shape)
    shape[1] *= repeats
    return x.new_empty(shape)


def gqa_repeat(x: Tensor, repeats: int) -> Tensor:
    return torch.ops.torch2iron.gqa_repeat.default(x, repeats)


@torch.library.custom_op("torch2iron::llama_chunked_attention", mutates_args=())
def _llama_chunked_attention_impl(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    packet_cache: Tensor,
) -> Tensor:
    del keys, values, packet_cache
    return torch.zeros_like(queries)


@_llama_chunked_attention_impl.register_fake
def _(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    packet_cache: Tensor,
) -> Tensor:
    del keys, values, packet_cache
    return queries.new_empty(queries.shape)


def llama_chunked_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    packet_cache: Tensor,
) -> Tensor:
    return torch.ops.torch2iron.llama_chunked_attention.default(
        queries, keys, values, packet_cache
    )


@dataclass(frozen=True)
class LlamaExportConfig:
    vocab_size: int = 128
    emb_dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_groups: int = 2
    head_dim: int = 8
    hidden_dim: int = 64
    max_seq_len: int = 8
    chunk_size: int = 4
    rms_norm_eps: float = 1e-5

    @property
    def q_heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_groups

    @property
    def packet_cache_elements(self) -> int:
        num_chunks = self.max_seq_len // self.chunk_size
        chunk_elements = 2 * self.chunk_size * self.head_dim + self.chunk_size
        return self.n_kv_groups * num_chunks * chunk_elements


class ExportRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, eps=self.eps)


class ExportLlamaMLP(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(swiglu(gate, up))


class ExportPrefillAttention(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.emb_dim, config.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.emb_dim, config.n_kv_groups * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.emb_dim, config.n_kv_groups * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.emb_dim, bias=False
        )

    def forward(self, x: Tensor, rope_angles: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        cfg = self.config

        queries = self.q_proj(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim)
        keys = self.k_proj(x).view(batch, seq_len, cfg.n_kv_groups, cfg.head_dim)
        values = self.v_proj(x).view(batch, seq_len, cfg.n_kv_groups, cfg.head_dim)

        queries = rope(queries, rope_angles)
        keys = rope(keys, rope_angles)

        query_heads = queries.transpose(1, 2)
        key_heads = gqa_repeat(keys.transpose(1, 2), cfg.q_heads_per_group)
        value_heads = gqa_repeat(values.transpose(1, 2), cfg.q_heads_per_group)

        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        scores = scores * (cfg.head_dim**-0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value_heads)
        context = context.transpose(1, 2).reshape(
            batch, seq_len, cfg.n_heads * cfg.head_dim
        )
        return self.o_proj(context), keys, values


class ExportChunkedDecodeAttention(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.emb_dim, config.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.emb_dim, config.n_kv_groups * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.emb_dim, config.n_kv_groups * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.emb_dim, bias=False
        )

    def forward(
        self,
        x: Tensor,
        rope_angles: Tensor,
        packet_cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        cfg = self.config

        queries = self.q_proj(x).view(
            batch,
            seq_len,
            cfg.n_kv_groups,
            cfg.q_heads_per_group,
            cfg.head_dim,
        )
        keys = self.k_proj(x).view(batch, seq_len, cfg.n_kv_groups, cfg.head_dim)
        values = self.v_proj(x).view(batch, seq_len, cfg.n_kv_groups, cfg.head_dim)

        queries = rope(queries, rope_angles)
        keys = rope(keys, rope_angles)

        context = llama_chunked_attention(queries, keys, values, packet_cache)
        context = context.reshape(batch, seq_len, cfg.n_heads * cfg.head_dim)
        return self.o_proj(context), keys, values


class ExportLlamaPrefillLayer(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.input_layernorm = ExportRMSNorm(config.emb_dim, config.rms_norm_eps)
        self.self_attn = ExportPrefillAttention(config)
        self.post_attention_layernorm = ExportRMSNorm(
            config.emb_dim, config.rms_norm_eps
        )
        self.mlp = ExportLlamaMLP(config)

    def forward(self, hidden_states: Tensor, rope_angles: Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key, present_value = self.self_attn(
            hidden_states, rope_angles
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present_key, present_value


class ExportLlamaDecodeLayer(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.input_layernorm = ExportRMSNorm(config.emb_dim, config.rms_norm_eps)
        self.self_attn = ExportChunkedDecodeAttention(config)
        self.post_attention_layernorm = ExportRMSNorm(
            config.emb_dim, config.rms_norm_eps
        )
        self.mlp = ExportLlamaMLP(config)

    def forward(self, hidden_states: Tensor, rope_angles: Tensor, packet_cache: Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key, present_value = self.self_attn(
            hidden_states, rope_angles, packet_cache
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present_key, present_value


class ExportLlamaPrefillModel(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ExportLlamaPrefillLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = ExportRMSNorm(config.emb_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x: Tensor, rope_angles: Tensor):
        present_keys = []
        present_values = []
        for layer in self.layers:
            x, present_key, present_value = layer(x, rope_angles)
            present_keys.append(present_key)
            present_values.append(present_value)

        logits = self.lm_head(self.norm(x))
        return (logits, *present_keys, *present_values)


class ExportLlamaDecodeModel(nn.Module):
    def __init__(self, config: LlamaExportConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ExportLlamaDecodeLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = ExportRMSNorm(config.emb_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x: Tensor, rope_angles: Tensor, *packet_caches: Tensor):
        if len(packet_caches) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} packet caches, got {len(packet_caches)}"
            )

        present_keys = []
        present_values = []
        for layer, packet_cache in zip(self.layers, packet_caches):
            x, present_key, present_value = layer(x, rope_angles, packet_cache)
            present_keys.append(present_key)
            present_values.append(present_value)

        logits = self.lm_head(self.norm(x))
        return (logits, *present_keys, *present_values)


def example_prefill_args(config: LlamaExportConfig) -> tuple[Tensor, ...]:
    x = torch.randn(1, config.max_seq_len, config.emb_dim)
    rope_angles = torch.randn(config.max_seq_len, config.head_dim)
    return (x, rope_angles)


def example_decode_args(config: LlamaExportConfig) -> tuple[Tensor, ...]:
    x = torch.randn(1, 1, config.emb_dim)
    rope_angles = torch.randn(1, config.head_dim)
    packet_caches = tuple(
        torch.randn(config.packet_cache_elements) for _ in range(config.n_layers)
    )
    return (x, rope_angles, *packet_caches)
