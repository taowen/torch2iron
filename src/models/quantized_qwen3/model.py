#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference runtime for inference-packed W4A16 Qwen3 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.quantized_qwen3.packed_format import (
    PackedInferenceStore,
    find_packed_dir,
)


def find_model_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path.parent
    if (path / "config.json").exists() and find_packed_dir(path) is not None:
        return path
    if (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists():
        return path
    if path.name == "qwen3_w4a16_packed" and (path.parent / "config.json").exists():
        return path.parent
    candidates = sorted(
        [
            manifest.parent.parent
            for manifest in path.rglob("qwen3_w4a16_packed/manifest.json")
            if (manifest.parent.parent / "config.json").exists()
        ],
        key=lambda candidate: (
            candidate / "qwen3_w4a16_packed" / "manifest.json"
        ).stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            [candidate.parent for candidate in path.rglob("model.safetensors")],
            key=lambda candidate: (candidate / "model.safetensors").stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"no Qwen3 packed model directory found under {path}")
    return candidates[0]


def _as_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return aliases[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}") from exc


class DenseLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, compute_dtype: torch.dtype):
        super().__init__()
        self.register_buffer("weight", weight.to(compute_dtype).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class PackedW4A16Linear(nn.Module):
    def __init__(
        self,
        *,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.compute_dtype = compute_dtype
        self.register_buffer("packed_weight", packed_weight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("_signed_weight", torch.empty(0, dtype=torch.int8), persistent=False)

    @classmethod
    def from_store(
        cls,
        store: PackedInferenceStore,
        prefix: str,
        *,
        compute_dtype: torch.dtype,
    ) -> "PackedW4A16Linear":
        spec, packed, scales = store.linear_segments(prefix)
        return cls(
            packed_weight=packed,
            scales=scales,
            in_features=int(spec["in_features"]),
            out_features=int(spec["out_features"]),
            group_size=int(spec["group_size"]),
            compute_dtype=compute_dtype,
        )

    def unpack_signed_weight(self) -> torch.Tensor:
        if self._signed_weight.numel() != 0:
            return self._signed_weight

        low = torch.bitwise_and(self.packed_weight, 0xF)
        high = torch.bitwise_and(torch.bitwise_right_shift(self.packed_weight, 4), 0xF)
        nibbles = torch.stack((low, high), dim=-1).flatten(1)[:, : self.in_features]
        signed = (nibbles.to(torch.int16) - 8).to(torch.int8)
        self._signed_weight = signed.contiguous()
        return self._signed_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signed = self.unpack_signed_weight()
        x_flat = x.reshape(-1, self.in_features).to(torch.float32)
        out = torch.zeros((x_flat.shape[0], self.out_features), dtype=torch.float32)
        scales = self.scales.to(torch.float32)
        for group_idx in range(scales.shape[1]):
            start = group_idx * self.group_size
            end = min(start + self.group_size, self.in_features)
            weight_group = (
                signed[:, start:end].to(torch.float32)
                * scales[:, group_idx].view(self.out_features, 1)
            )
            out += x_flat[:, start:end].matmul(weight_group.t())
        return out.reshape(*x.shape[:-1], self.out_features).to(self.compute_dtype)


class PackedTensorWeights:
    def __init__(self, store: PackedInferenceStore) -> None:
        self.store = store

    def dense(self, name: str) -> torch.Tensor:
        return self.store.dense(name)

    def has_linear(self, prefix: str) -> bool:
        return self.store.has_linear(prefix)

    def linear(
        self,
        prefix: str,
        *,
        compute_dtype: torch.dtype,
    ) -> nn.Module:
        return PackedW4A16Linear.from_store(
            self.store,
            prefix,
            compute_dtype=compute_dtype,
        )


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float, compute_dtype: torch.dtype):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("weight", weight.to(compute_dtype).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cos = angles[..., ::2].repeat_interleave(2, dim=-1)
    sin = angles[..., 1::2].repeat_interleave(2, dim=-1)
    view_shape = [1, angles.shape[-2], *([1] * (x.dim() - 3)), angles.shape[-1]]
    cos = cos.view(*view_shape)
    sin = sin.view(*view_shape)
    return x * cos.to(x.dtype) + _rotate_half(x) * sin.to(x.dtype)


class QuantizedQwenMLP(nn.Module):
    def __init__(
        self,
        weights,
        prefix: str,
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        kwargs = {
            "compute_dtype": compute_dtype,
        }
        self.gate_proj = weights.linear(f"{prefix}.gate_proj", **kwargs)
        self.up_proj = weights.linear(f"{prefix}.up_proj", **kwargs)
        self.down_proj = weights.linear(f"{prefix}.down_proj", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QuantizedQwenAttention(nn.Module):
    def __init__(
        self,
        weights,
        prefix: str,
        config: "QuantizedQwenConfig",
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        kwargs = {
            "compute_dtype": compute_dtype,
        }
        self.q_proj = weights.linear(f"{prefix}.q_proj", **kwargs)
        self.k_proj = weights.linear(f"{prefix}.k_proj", **kwargs)
        self.v_proj = weights.linear(f"{prefix}.v_proj", **kwargs)
        self.o_proj = weights.linear(f"{prefix}.o_proj", **kwargs)
        self.q_norm = RMSNorm(weights.dense(f"{prefix}.q_norm.weight"), config.rms_norm_eps, compute_dtype)
        self.k_norm = RMSNorm(weights.dense(f"{prefix}.k_norm.weight"), config.rms_norm_eps, compute_dtype)

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape
        cfg = self.config
        q = self.q_proj(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim)
        k = self.k_proj(x).view(batch, seq_len, cfg.n_kv_groups, cfg.head_dim)
        v = self.v_proj(x).view(batch, seq_len, cfg.n_kv_groups, cfg.head_dim)

        q = apply_rope(self.q_norm(q), angles)
        k = apply_rope(self.k_norm(k), angles)

        q_heads = q.transpose(1, 2)
        if past_key_value is None:
            past_len = 0
            key_states = k.transpose(1, 2)
            value_states = v.transpose(1, 2)
        else:
            past_len = past_key_value[0].shape[2]
            key_states = torch.cat([past_key_value[0], k.transpose(1, 2)], dim=2)
            value_states = torch.cat([past_key_value[1], v.transpose(1, 2)], dim=2)
        present = (key_states, value_states)

        key_heads = key_states.repeat_interleave(cfg.q_heads_per_group, dim=1)
        value_heads = value_states.repeat_interleave(cfg.q_heads_per_group, dim=1)
        scores = torch.matmul(q_heads.to(torch.float32), key_heads.transpose(-2, -1).to(torch.float32))
        scores *= cfg.head_dim**-0.5

        total_len = key_states.shape[2]
        query_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
        key_positions = torch.arange(total_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, total_len), torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1).to(x.dtype)
        context = torch.matmul(weights, value_heads).transpose(1, 2)
        context = context.reshape(batch, seq_len, cfg.n_heads * cfg.head_dim)
        return self.o_proj(context), present


class QuantizedQwenLayer(nn.Module):
    def __init__(
        self,
        weights,
        layer_idx: int,
        config: "QuantizedQwenConfig",
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = RMSNorm(
            weights.dense(f"{prefix}.input_layernorm.weight"),
            config.rms_norm_eps,
            compute_dtype,
        )
        self.self_attn = QuantizedQwenAttention(
            weights,
            f"{prefix}.self_attn",
            config,
            compute_dtype=compute_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            weights.dense(f"{prefix}.post_attention_layernorm.weight"),
            config.rms_norm_eps,
            compute_dtype,
        )
        self.mlp = QuantizedQwenMLP(
            weights,
            f"{prefix}.mlp",
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        attn_output, present = self.self_attn(self.input_layernorm(x), angles, past_key_value)
        x = residual + attn_output
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x, present


@dataclass(frozen=True)
class QuantizedQwenConfig:
    vocab_size: int
    emb_dim: int
    n_layers: int
    n_heads: int
    n_kv_groups: int
    head_dim: int
    hidden_dim: int
    rms_norm_eps: float
    rope_base: float
    context_length: int
    eos_token_id: int | list[int] | None
    bos_token_id: int | None
    group_size: int

    @property
    def q_heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_groups


def compute_rope_angles(head_dim: int, context_length: int, rope_base: float) -> torch.Tensor:
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    position = torch.arange(context_length).float()
    freqs = torch.outer(position, inv_freq)
    angles = torch.empty(context_length, head_dim)
    angles[:, ::2] = torch.cos(freqs)
    angles[:, 1::2] = torch.sin(freqs)
    return angles


class QuantizedQwenForCausalLM(nn.Module):
    def __init__(
        self,
        weights,
        hf_config: dict[str, object],
        *,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        quant_config = hf_config.get("quantization_config") or {}
        bits = int(quant_config.get("bits", 4))
        if bits != 4:
            raise ValueError(f"only W4A16 checkpoints are supported, got bits={bits}")
        if not bool(quant_config.get("sym", True)):
            raise ValueError("only symmetric AutoGPTQ W4A16 checkpoints are supported")
        group_size = int(quant_config.get("group_size", 128))

        self.config = QuantizedQwenConfig(
            vocab_size=int(hf_config["vocab_size"]),
            emb_dim=int(hf_config["hidden_size"]),
            n_layers=int(hf_config["num_hidden_layers"]),
            n_heads=int(hf_config["num_attention_heads"]),
            n_kv_groups=int(hf_config["num_key_value_heads"]),
            head_dim=int(hf_config.get("head_dim", int(hf_config["hidden_size"]) // int(hf_config["num_attention_heads"]))),
            hidden_dim=int(hf_config["intermediate_size"]),
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_base=float((hf_config.get("rope_parameters") or {}).get("rope_theta", hf_config.get("rope_theta", 1000000.0))),
            context_length=int(hf_config.get("max_position_embeddings", 40960)),
            eos_token_id=hf_config.get("eos_token_id"),
            bos_token_id=hf_config.get("bos_token_id"),
            group_size=group_size,
        )
        self.compute_dtype = compute_dtype
        self.embed_tokens = nn.Embedding.from_pretrained(
            weights.dense("model.embed_tokens.weight").to(compute_dtype),
            freeze=True,
        )
        self.layers = nn.ModuleList(
            [
                QuantizedQwenLayer(
                    weights,
                    layer_idx,
                    self.config,
                    compute_dtype=compute_dtype,
                )
                for layer_idx in range(self.config.n_layers)
            ]
        )
        self.norm = RMSNorm(weights.dense("model.norm.weight"), self.config.rms_norm_eps, compute_dtype)
        if weights.has_linear("lm_head"):
            self.lm_head = weights.linear(
                "lm_head",
                compute_dtype=compute_dtype,
            )
        else:
            try:
                lm_head_weight = weights.dense("lm_head.weight")
            except KeyError:
                lm_head_weight = weights.dense("model.embed_tokens.weight")
            self.lm_head = DenseLinear(lm_head_weight, compute_dtype)
        self.register_buffer(
            "rope_angles",
            compute_rope_angles(self.config.head_dim, self.config.context_length, self.config.rope_base).to(compute_dtype),
            persistent=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        compute_dtype: str | torch.dtype = torch.float32,
    ) -> tuple["QuantizedQwenForCausalLM", AutoTokenizer]:
        model_dir = find_model_dir(model_path)
        hf_config = json.loads((model_dir / "config.json").read_text())
        packed_dir = find_packed_dir(model_dir)
        if packed_dir is None:
            raise FileNotFoundError(
                f"missing inference-packed W4A16 artifact under {model_dir}; "
                "run `python -m models.quantized_qwen3.pack` first"
            )
        weights = PackedTensorWeights(PackedInferenceStore(packed_dir))
        model = cls(
            weights,
            hf_config,
            compute_dtype=_as_torch_dtype(compute_dtype),
        )
        model.weight_source = "packed"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        return model.eval(), tokenizer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if batch != 1:
            raise ValueError("reference runtime currently supports batch size 1")
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_len + seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length {past_len + seq_len} exceeds context {self.config.context_length}"
            )

        x = self.embed_tokens(input_ids).to(self.compute_dtype)
        angles = self.rope_angles[past_len : past_len + seq_len]
        present_key_values = []
        if past_key_values is None:
            past_key_values = (None,) * len(self.layers)
        for layer, past in zip(self.layers, past_key_values):
            x, present = layer(x, angles, past)
            present_key_values.append(present)
        logits = self.lm_head(self.norm(x))
        return logits, tuple(present_key_values)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        logits, past = self.forward(input_ids)
        eos = set(eos_token_ids or [])
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            token_id = int(next_token.item())
            if token_id in eos:
                break
            logits, past = self.forward(next_token, past)
        return generated
