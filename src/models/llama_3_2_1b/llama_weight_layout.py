# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


PACKED_WEIGHTS_FORMAT = "llama_3_2_1b_iron_packed_weights_v1"
PACKED_WEIGHTS_BIN = "weights.bf16.bin"
PACKED_WEIGHTS_MANIFEST = "manifest.json"

LAYER_WEIGHT_ORDER = (
    ("W_norm1_{layer}", "model.layers.{layer}.input_layernorm.weight"),
    ("W_attn_query_{layer}", "model.layers.{layer}.self_attn.q_proj.weight"),
    ("W_attn_key_{layer}", "model.layers.{layer}.self_attn.k_proj.weight"),
    ("W_attn_value_{layer}", "model.layers.{layer}.self_attn.v_proj.weight"),
    ("W_attn_output_decode_{layer}", "model.layers.{layer}.self_attn.o_proj.weight"),
    ("W_norm2_{layer}", "model.layers.{layer}.post_attention_layernorm.weight"),
    ("W_ffn_gate_{layer}", "model.layers.{layer}.mlp.gate_proj.weight"),
    ("W_ffn_up_{layer}", "model.layers.{layer}.mlp.up_proj.weight"),
    ("W_ffn_down_{layer}", "model.layers.{layer}.mlp.down_proj.weight"),
)

GLOBAL_WEIGHT_ORDER = (
    ("W_final_norm", "model.norm.weight", "weight"),
    ("W_out_head", "model.embed_tokens.weight", "lm_head"),
)


def default_llama_packed_weights_dir(weights_path: str | Path) -> Path:
    return Path(weights_path).resolve().parent / "llama_iron_packed"


def _config_dict(config) -> dict[str, object]:
    keys = (
        "vocab_size",
        "emb_dim",
        "n_layers",
        "n_heads",
        "n_kv_groups",
        "head_dim",
        "hidden_dim",
        "rope_base",
        "context_length",
    )
    return {key: getattr(config, key) for key in keys}


def _byte_alignment(byte_offset: int) -> int | None:
    if byte_offset == 0:
        return None
    return byte_offset & -byte_offset


def _write_bf16_tensor(file_obj, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    raw = tensor.flatten().contiguous().view(torch.uint16).cpu().numpy()
    raw.tofile(file_obj)


def _segment_entry(
    *,
    name: str,
    source: str,
    group: str,
    shape: torch.Size,
    element_offset: int,
    numel: int,
) -> dict[str, object]:
    byte_offset = element_offset * 2
    return {
        "name": name,
        "source": source,
        "group": group,
        "shape": list(shape),
        "dtype": "bfloat16",
        "element_offset": element_offset,
        "numel": numel,
        "byte_offset": byte_offset,
        "byte_length": numel * 2,
        "byte_alignment": _byte_alignment(byte_offset),
    }


def iter_llama_decode_weight_specs(config):
    for layer_idx in range(config.n_layers):
        for buffer_template, source_template in LAYER_WEIGHT_ORDER:
            yield {
                "layer": layer_idx,
                "group": "weight",
                "name": buffer_template.format(layer=layer_idx),
                "source": source_template.format(layer=layer_idx),
            }
    for name, source, group in GLOBAL_WEIGHT_ORDER:
        yield {
            "layer": None,
            "group": group,
            "name": name,
            "source": source,
        }


def llama_decode_transformer_weight_names(config) -> list[str]:
    return [
        spec["name"]
        for spec in iter_llama_decode_weight_specs(config)
        if spec["group"] == "weight"
    ]


def llama_decode_lm_head_weight_names(config) -> list[str]:
    return [
        spec["name"]
        for spec in iter_llama_decode_weight_specs(config)
        if spec["group"] == "lm_head"
    ]


def build_llama_packed_weights_manifest(config) -> dict[str, object]:
    offset = 0
    layers: list[dict[str, object]] = []
    flat_segments: list[dict[str, object]] = []

    for layer_idx in range(config.n_layers):
        layer_offset = offset
        layer_segments = []
        for buffer_template, source_template in LAYER_WEIGHT_ORDER:
            name = buffer_template.format(layer=layer_idx)
            source = source_template.format(layer=layer_idx)
            tensor = config.weights[source]
            numel = tensor.numel()
            segment = _segment_entry(
                name=name,
                source=source,
                group="weight",
                shape=tensor.shape,
                element_offset=offset,
                numel=numel,
            )
            layer_segments.append(segment)
            flat_segments.append(segment)
            offset += numel
        layers.append(
            {
                "id": layer_idx,
                "group": "weight",
                "element_offset": layer_offset,
                "numel": offset - layer_offset,
                "byte_offset": layer_offset * 2,
                "byte_length": (offset - layer_offset) * 2,
                "byte_alignment": _byte_alignment(layer_offset * 2),
                "segments": layer_segments,
            }
        )

    globals_ = []
    for name, source, group in GLOBAL_WEIGHT_ORDER:
        tensor = config.weights[source]
        numel = tensor.numel()
        segment = _segment_entry(
            name=name,
            source=source,
            group=group,
            shape=tensor.shape,
            element_offset=offset,
            numel=numel,
        )
        globals_.append(segment)
        flat_segments.append(segment)
        offset += numel

    group_ranges = {}
    for group in ("weight", "lm_head"):
        group_segments = [seg for seg in flat_segments if seg["group"] == group]
        if group_segments:
            start = int(group_segments[0]["element_offset"])
            end = int(group_segments[-1]["element_offset"]) + int(
                group_segments[-1]["numel"]
            )
        else:
            start = end = 0
        group_ranges[group] = {
            "element_offset": start,
            "numel": end - start,
            "byte_offset": start * 2,
            "byte_length": (end - start) * 2,
        }

    return {
        "format": PACKED_WEIGHTS_FORMAT,
        "dtype": "bfloat16",
        "element_size_bytes": 2,
        "weight_order": [spec["name"] for spec in iter_llama_decode_weight_specs(config)],
        "model_config": _config_dict(config),
        "num_layers": config.n_layers,
        "total_numel": offset,
        "total_bytes": offset * 2,
        "data_file": PACKED_WEIGHTS_BIN,
        "groups": group_ranges,
        "layers": layers,
        "globals": globals_,
        "segments": flat_segments,
    }


def write_llama_packed_weight_artifact(config, output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_llama_packed_weights_manifest(config)
    tmp_bin = output_dir / f"{PACKED_WEIGHTS_BIN}.tmp"
    tmp_manifest = output_dir / f"{PACKED_WEIGHTS_MANIFEST}.tmp"
    bin_path = output_dir / PACKED_WEIGHTS_BIN
    manifest_path = output_dir / PACKED_WEIGHTS_MANIFEST

    with open(tmp_bin, "wb") as f:
        for segment in manifest["segments"]:
            _write_bf16_tensor(f, config.weights[segment["source"]])
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_bin.replace(bin_path)
    tmp_manifest.replace(manifest_path)
    validate_llama_packed_weight_artifact(config, output_dir)
    return manifest


def load_llama_packed_weights_manifest(packed_dir: str | Path) -> dict[str, object]:
    manifest_path = Path(packed_dir) / PACKED_WEIGHTS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing packed weight manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def validate_llama_packed_weight_artifact(
    config, packed_dir: str | Path
) -> dict[str, object]:
    packed_dir = Path(packed_dir)
    manifest = load_llama_packed_weights_manifest(packed_dir)
    if manifest.get("format") != PACKED_WEIGHTS_FORMAT:
        raise RuntimeError(
            f"unsupported packed weight format {manifest.get('format')!r}; "
            f"expected {PACKED_WEIGHTS_FORMAT}"
        )
    if manifest.get("dtype") != "bfloat16":
        raise RuntimeError(f"packed weight dtype must be bfloat16: {manifest.get('dtype')}")
    if manifest.get("model_config") != _config_dict(config):
        raise RuntimeError("packed weight model_config does not match current config")
    expected_names = [spec["name"] for spec in iter_llama_decode_weight_specs(config)]
    if manifest.get("weight_order") != expected_names:
        raise RuntimeError("packed weight order does not match decode layout")
    if int(manifest.get("num_layers", -1)) != config.n_layers:
        raise RuntimeError("packed weight layer count mismatch")

    segments = manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != len(expected_names):
        raise RuntimeError("packed weight segment table is missing or incomplete")
    offset = 0
    for spec, segment in zip(iter_llama_decode_weight_specs(config), segments):
        tensor = config.weights[spec["source"]]
        if segment["name"] != spec["name"]:
            raise RuntimeError(f"packed segment name mismatch for {spec['name']}")
        if segment["source"] != spec["source"]:
            raise RuntimeError(f"packed segment source mismatch for {spec['name']}")
        if segment["shape"] != list(tensor.shape):
            raise RuntimeError(f"packed segment shape mismatch for {spec['name']}")
        if int(segment["element_offset"]) != offset:
            raise RuntimeError(f"packed segment offset mismatch for {spec['name']}")
        if int(segment["numel"]) != tensor.numel():
            raise RuntimeError(f"packed segment size mismatch for {spec['name']}")
        if int(segment["byte_offset"]) % 64 != 0:
            raise RuntimeError(f"packed segment {spec['name']} is not 64B aligned")
        offset += tensor.numel()

    expected_bytes = int(manifest["total_bytes"])
    if offset * 2 != expected_bytes:
        raise RuntimeError("packed manifest total size mismatch")
    bin_path = packed_dir / PACKED_WEIGHTS_BIN
    if not bin_path.exists():
        raise FileNotFoundError(f"missing packed weight data file: {bin_path}")
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"packed weight file size {actual_bytes} bytes != expected {expected_bytes}"
        )
    return manifest


def find_llama_packed_segment(
    manifest: dict[str, object], name: str
) -> dict[str, object]:
    for segment in manifest["segments"]:
        if segment["name"] == name:
            return segment
    raise KeyError(f"missing packed segment {name!r}")


def load_llama_packed_segment(
    packed_dir: str | Path, manifest: dict[str, object], name: str
) -> torch.Tensor:
    segment = find_llama_packed_segment(manifest, name)
    path = Path(packed_dir) / manifest["data_file"]
    raw = np.fromfile(
        path,
        dtype=np.uint16,
        count=int(segment["numel"]),
        offset=int(segment["byte_offset"]),
    )
    if raw.size != int(segment["numel"]):
        raise RuntimeError(f"short read for packed segment {name!r}")
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)
