# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.exported_qwen3.qwen_weight_layout import iter_qwen_decode_weight_specs


PACKED_WEIGHTS_FORMAT = "exported_qwen3_iron_packed_weights_v1"
LEGACY_PACKED_WEIGHTS_FORMATS = ()
PACKED_WEIGHTS_BIN = "weights.bf16.bin"
PACKED_WEIGHTS_MANIFEST = "manifest.json"


def default_qwen_packed_weights_dir(weights_path: str | Path) -> Path:
    return Path(weights_path).resolve().parent / "qwen_iron_packed"


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


def build_qwen_packed_weights_manifest(config) -> dict[str, object]:
    offset = 0
    layers: list[dict[str, object]] = []
    flat_segments: list[dict[str, object]] = []

    generated_specs = list(iter_qwen_decode_weight_specs(config))
    for layer_idx in range(config.n_layers):
        layer_offset = offset
        layer_segments = []
        for spec in generated_specs:
            if spec["layer"] != layer_idx or spec["group"] != "weight":
                continue
            name = spec["name"]
            source = spec["source"]
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
    for spec in generated_specs:
        if spec["layer"] is not None:
            continue
        name = spec["name"]
        source = spec["source"]
        group = spec["group"]
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
        "weight_order": [spec["name"] for spec in iter_qwen_decode_weight_specs(config)],
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


def write_qwen_packed_weight_artifact(config, output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_qwen_packed_weights_manifest(config)
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
    validate_qwen_packed_weight_artifact(config, output_dir)
    return manifest


def load_qwen_packed_weights_manifest(packed_dir: str | Path) -> dict[str, object]:
    manifest_path = Path(packed_dir) / PACKED_WEIGHTS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing packed weight manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def validate_qwen_packed_weight_artifact(
    config, packed_dir: str | Path
) -> dict[str, object]:
    packed_dir = Path(packed_dir)
    manifest = load_qwen_packed_weights_manifest(packed_dir)
    expected = build_qwen_packed_weights_manifest(config)
    supported_formats = (PACKED_WEIGHTS_FORMAT,) + LEGACY_PACKED_WEIGHTS_FORMATS
    if manifest.get("format") not in supported_formats:
        raise RuntimeError(
            f"unsupported packed weight format {manifest.get('format')!r}; "
            f"expected one of {supported_formats}"
        )

    for key in (
        "dtype",
        "element_size_bytes",
        "weight_order",
        "model_config",
        "num_layers",
        "total_numel",
        "total_bytes",
        "data_file",
        "groups",
    ):
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"packed manifest {key} does not match current config")

    expected_segments = expected["segments"]
    segments = manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != len(expected_segments):
        raise RuntimeError("packed weight segment table is missing or incomplete")
    for segment, expected_segment in zip(segments, expected_segments):
        for key in ("name", "source", "shape", "element_offset", "numel", "byte_offset"):
            if segment.get(key) != expected_segment[key]:
                raise RuntimeError(
                    f"packed segment {key} mismatch for {expected_segment['name']}"
                )
        if int(segment["byte_offset"]) % 64 != 0:
            raise RuntimeError(f"packed segment {segment['name']} is not 64B aligned")

    bin_path = packed_dir / PACKED_WEIGHTS_BIN
    if not bin_path.exists():
        raise FileNotFoundError(f"missing packed weight data file: {bin_path}")
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected["total_bytes"]:
        raise RuntimeError(
            f"packed weight file size {actual_bytes} bytes "
            f"!= expected {expected['total_bytes']}"
        )
    return manifest


def load_qwen_packed_segment(
    packed_dir: str | Path, manifest: dict[str, object], name: str
) -> torch.Tensor:
    try:
        segment = next(seg for seg in manifest["segments"] if seg["name"] == name)
    except StopIteration as exc:
        raise KeyError(f"missing packed segment {name!r}") from exc
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
