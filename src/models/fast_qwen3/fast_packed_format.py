#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast Qwen3 packed artifact for fused online-Q4 layer engines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.quantized_qwen3.model import find_model_dir
from models.quantized_qwen3.packed_format import (
    unpack_autogptq_qweight,
    unpack_autogptq_qzeros,
)
from models.quantized_qwen3.packed_format import _load_safetensors as load_safetensors

from models.fast_qwen3.q4nx_layout import (
    Q4NXLinearSpec,
    compression_ratio,
    linear_total_bytes,
    manifest_entry,
    pack_linear_weight,
    q4nx_linear_reference,
    spec_from_manifest,
)


FAST_QWEN3_FORMAT = "fast_qwen3_q4nx_layer_v1"
FAST_QWEN3_DIRNAME = "fast_qwen3_q4nx"
FAST_QWEN3_MANIFEST = "manifest.json"
FAST_QWEN3_DATA = "weights.q4nx.bin"
SEGMENT_ALIGNMENT = 64


def default_fast_dir(model_dir: str | Path) -> Path:
    return Path(model_dir) / FAST_QWEN3_DIRNAME


def find_fast_dir(path: str | Path) -> Path | None:
    path = Path(path).expanduser().resolve()
    if _is_fast_manifest(path / FAST_QWEN3_MANIFEST):
        return path
    candidate = path / FAST_QWEN3_DIRNAME
    if _is_fast_manifest(candidate / FAST_QWEN3_MANIFEST):
        return candidate
    candidates = sorted(
        [
            manifest.parent
            for manifest in path.rglob(FAST_QWEN3_MANIFEST)
            if _is_fast_manifest(manifest)
        ],
        key=lambda manifest_dir: (manifest_dir / FAST_QWEN3_MANIFEST).stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _is_fast_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("format") == FAST_QWEN3_FORMAT


def _align(file_obj) -> None:
    padding = (-file_obj.tell()) % SEGMENT_ALIGNMENT
    if padding:
        file_obj.write(b"\0" * padding)


def _write_segment(file_obj, data: bytes) -> tuple[int, int]:
    _align(file_obj)
    offset = file_obj.tell()
    file_obj.write(data)
    return offset, len(data)


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16)
    return raw.numpy().tobytes()


def _dense_segment(name: str, tensor: torch.Tensor, offset: int, length: int) -> dict:
    return {
        "name": name,
        "layout": "contiguous_bf16",
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": "bfloat16",
        "byte_offset": offset,
        "byte_length": length,
    }


def _linear_prefixes(tensors: dict[str, torch.Tensor]) -> list[str]:
    qweight_prefixes = [name[: -len(".qweight")] for name in tensors if name.endswith(".qweight")]
    if qweight_prefixes:
        return sorted(qweight_prefixes, key=_qwen_linear_sort_key)
    dense_prefixes = [
        name[: -len(".weight")]
        for name, tensor in tensors.items()
        if name.endswith(".weight") and tensor.dim() == 2
    ]
    return sorted(dense_prefixes, key=_qwen_linear_sort_key)


def _qwen_linear_sort_key(prefix: str) -> tuple[int, int, str]:
    marker = ".layers."
    order = {
        "self_attn.q_proj": 0,
        "self_attn.k_proj": 1,
        "self_attn.v_proj": 2,
        "self_attn.o_proj": 3,
        "mlp.up_proj": 4,
        "mlp.gate_proj": 5,
        "mlp.down_proj": 6,
    }
    if marker in prefix:
        tail = prefix.split(marker, 1)[1]
        layer_text, suffix = tail.split(".", 1)
        layer_idx = int(layer_text)
        return (layer_idx, order.get(suffix, 100), suffix)
    if prefix == "lm_head":
        return (1_000_000, 0, prefix)
    return (900_000, 0, prefix)


def _autogptq_linear_to_dense(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    *,
    group_size: int,
) -> torch.Tensor:
    qweight = tensors[f"{prefix}.qweight"]
    qzeros = tensors[f"{prefix}.qzeros"]
    scales = tensors[f"{prefix}.scales"].detach().cpu().to(torch.float32).contiguous()
    in_features = int(qweight.shape[0] * 8)
    out_features = int(qweight.shape[1])
    qvalues = unpack_autogptq_qweight(qweight, in_features, out_features).to(torch.float32)
    zeros = unpack_autogptq_qzeros(qzeros, out_features).to(torch.float32)

    if scales.shape[0] != zeros.shape[0] and scales.shape[1] == zeros.shape[0]:
        scales = scales.t().contiguous()
    if scales.shape != zeros.shape:
        raise ValueError(
            f"scale/zero shape mismatch for {prefix}: "
            f"scales={tuple(scales.shape)} zeros={tuple(zeros.shape)}"
        )

    dense = torch.empty((out_features, in_features), dtype=torch.float32)
    for group_idx in range(scales.shape[0]):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        dense[:, start:end] = (
            qvalues[:, start:end] - zeros[group_idx].view(out_features, 1)
        ) * scales[group_idx].view(out_features, 1)
    return dense


def _linear_weight(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    *,
    group_size: int,
) -> torch.Tensor:
    if f"{prefix}.qweight" in tensors:
        return _autogptq_linear_to_dense(tensors, prefix, group_size=group_size)
    weight_name = f"{prefix}.weight"
    if weight_name not in tensors:
        raise KeyError(f"missing linear weight for {prefix}")
    return tensors[weight_name].detach().cpu().to(torch.float32).contiguous()


def write_fast_qwen3_artifact(
    model_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    model_dir = find_model_dir(model_dir)
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else default_fast_dir(model_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Qwen config: {config_path}")
    config = json.loads(config_path.read_text())
    quant_config = config.get("quantization_config") or {}
    source_group_size = int(quant_config.get("group_size", 128))
    tensors = load_safetensors(model_dir)
    linear_prefixes = _linear_prefixes(tensors)

    tmp_data = output_dir / f"{FAST_QWEN3_DATA}.tmp"
    data_path = output_dir / FAST_QWEN3_DATA
    tmp_manifest = output_dir / f"{FAST_QWEN3_MANIFEST}.tmp"
    manifest_path = output_dir / FAST_QWEN3_MANIFEST

    manifest = {
        "format": FAST_QWEN3_FORMAT,
        "data_file": FAST_QWEN3_DATA,
        "source_model_dir": str(model_dir),
        "alignment": SEGMENT_ALIGNMENT,
        "q4nx": {
            "group_size": 32,
            "out_chunk": 32,
            "in_chunk": 256,
            "patch_out_rows": 64,
            "dequant": "weight = (q4 - zero_point) * scale",
        },
        "model_config": {
            "vocab_size": config.get("vocab_size"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "intermediate_size": config.get("intermediate_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rms_norm_eps": config.get("rms_norm_eps"),
        },
        "dense": {},
        "linears": {},
    }

    with open(tmp_data, "wb") as f:
        for name in sorted(tensors):
            if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
                continue
            if name.endswith(".weight") and name[: -len(".weight")] in linear_prefixes:
                continue
            tensor = tensors[name]
            offset, length = _write_segment(f, _bf16_bytes(tensor))
            manifest["dense"][name] = _dense_segment(name, tensor, offset, length)

        for prefix in linear_prefixes:
            weight = _linear_weight(tensors, prefix, group_size=source_group_size)
            packed_bytes, spec = pack_linear_weight(weight)
            offset, length = _write_segment(f, packed_bytes)
            entry = manifest_entry(spec, name=prefix, byte_offset=offset)
            entry["byte_length"] = length
            entry["source_group_size"] = source_group_size
            entry["bf16_compression_ratio"] = compression_ratio(
                int(weight.shape[1]),
                int(weight.shape[0]),
            )
            if length != linear_total_bytes(int(weight.shape[1]), int(weight.shape[0])):
                raise AssertionError(f"unexpected packed length for {prefix}")
            manifest["linears"][prefix] = entry

        _align(f)
        manifest["total_bytes"] = f.tell()

    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_data.replace(data_path)
    tmp_manifest.replace(manifest_path)
    return manifest


class FastQwen3Store:
    def __init__(self, packed_dir: str | Path) -> None:
        packed_dir = Path(packed_dir).expanduser().resolve()
        manifest_path = packed_dir / FAST_QWEN3_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing fast Qwen3 manifest: {manifest_path}")
        self.packed_dir = packed_dir
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != FAST_QWEN3_FORMAT:
            raise RuntimeError(
                f"unsupported fast Qwen3 format {self.manifest.get('format')!r}"
            )
        self.data_path = packed_dir / self.manifest["data_file"]
        self._data = np.memmap(self.data_path, dtype=np.uint8, mode="r")

    def dense(self, name: str) -> torch.Tensor:
        segment = self.manifest["dense"][name]
        raw = np.ndarray(
            shape=tuple(int(dim) for dim in segment["shape"]),
            dtype=np.uint16,
            buffer=self._data,
            offset=int(segment["byte_offset"]),
        )
        return torch.from_numpy(raw.copy()).view(torch.bfloat16)

    def linear_spec(self, prefix: str) -> Q4NXLinearSpec:
        return spec_from_manifest(prefix, self.manifest["linears"][prefix])

    def linear_bytes(self, prefix: str) -> memoryview:
        entry = self.manifest["linears"][prefix]
        offset = int(entry["byte_offset"])
        length = int(entry["byte_length"])
        return memoryview(self._data)[offset : offset + length]

    def linear_reference(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        return q4nx_linear_reference(x, self.linear_bytes(prefix), self.linear_spec(prefix))
