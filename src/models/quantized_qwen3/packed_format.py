#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-oriented W4A16 weight packing for Qwen3.

AutoRound/AutoGPTQ is a good interchange format, but its ``qweight/qzeros``
layout is not what we want to feed during inference. This module converts that
layout once into an aligned binary blob:

* dense tensors are stored as contiguous bf16 segments;
* quantized Linear weights are stored in the exact row-major qparam layout used
  by AIE: biased signed int4 values packed along K, followed by that row's bf16
  group scales.  The nibble value is ``signed + 8`` so AIE kernels can
  dequantize with an unpack and a subtract.

The fast AIE path can consume the same blob layout; the PyTorch runtime uses it
as a fused dequant/GEMM reference path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import torch


PACKED_FORMAT = "quantized_qwen3_w4a16_inference_v9"
PACKED_DIRNAME = "qwen3_w4a16_packed"
PACKED_MANIFEST = "manifest.json"
PACKED_DATA = "weights.w4a16.bin"
SEGMENT_ALIGNMENT = 64


def default_packed_dir(model_dir: str | Path) -> Path:
    return Path(model_dir) / PACKED_DIRNAME


def _is_supported_manifest(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("format") == PACKED_FORMAT
    except Exception:
        return False


def find_packed_dir(path: str | Path) -> Path | None:
    path = Path(path).expanduser().resolve()
    if (path / PACKED_MANIFEST).exists() and _is_supported_manifest(path / PACKED_MANIFEST):
        return path
    candidate = path / PACKED_DIRNAME
    if (candidate / PACKED_MANIFEST).exists() and _is_supported_manifest(candidate / PACKED_MANIFEST):
        return candidate
    candidates = sorted(
        [manifest.parent for manifest in path.rglob(PACKED_MANIFEST) if _is_supported_manifest(manifest)],
        key=lambda candidate_manifest: (candidate_manifest / PACKED_MANIFEST).stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_safetensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    single = model_dir / "model.safetensors"
    if single.exists():
        return safetensors.torch.load_file(single)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors file in {model_dir}")
    index = json.loads(index_path.read_text())
    filenames = sorted(set(index["weight_map"].values()))
    tensors: dict[str, torch.Tensor] = {}
    for filename in filenames:
        tensors.update(safetensors.torch.load_file(model_dir / filename))
    return tensors


def unpack_int32_nibbles(packed: torch.Tensor, values_per_word: int = 8) -> torch.Tensor:
    shifts = torch.arange(0, values_per_word * 4, 4, device=packed.device, dtype=torch.int32)
    unpacked = torch.bitwise_right_shift(packed.to(torch.int32).unsqueeze(-1), shifts)
    return torch.bitwise_and(unpacked, 0xF).to(torch.int16)


def unpack_autogptq_qweight(
    qweight: torch.Tensor,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    unpacked = unpack_int32_nibbles(qweight)
    unpacked = unpacked.permute(1, 0, 2).reshape(out_features, -1)
    return unpacked[:, :in_features].contiguous()


def unpack_autogptq_qzeros(qzeros: torch.Tensor, out_features: int) -> torch.Tensor:
    unpacked = unpack_int32_nibbles(qzeros).reshape(qzeros.shape[0], -1)
    return (unpacked[:, :out_features] + 1).contiguous()


def autogptq_to_signed_int4(
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    in_features: int,
    out_features: int,
    group_size: int,
) -> torch.Tensor:
    q_unsigned = unpack_autogptq_qweight(qweight, in_features, out_features)
    zeros = unpack_autogptq_qzeros(qzeros, out_features)
    signed = torch.empty_like(q_unsigned, dtype=torch.int16)
    for group_idx in range(math.ceil(in_features / group_size)):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        zero = zeros[group_idx].view(out_features, 1)
        signed[:, start:end] = q_unsigned[:, start:end] - zero

    min_q = int(signed.min().item())
    max_q = int(signed.max().item())
    if min_q < -8 or max_q > 7:
        raise ValueError(
            "AutoGPTQ weights do not fit signed int4 after zero-point removal: "
            f"range [{min_q}, {max_q}]"
        )
    return signed.to(torch.int8).contiguous()


def pack_signed_int4_bytes(signed: torch.Tensor) -> torch.Tensor:
    if signed.dtype != torch.int8:
        signed = signed.to(torch.int8)
    out_features, in_features = signed.shape
    if in_features % 2:
        signed = torch.cat([signed, torch.zeros(out_features, 1, dtype=torch.int8)], dim=1)
    nibbles = torch.bitwise_and(signed, 0xF).to(torch.uint8)
    low = nibbles[:, 0::2]
    high = torch.bitwise_left_shift(nibbles[:, 1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def _align(file_obj, alignment: int = SEGMENT_ALIGNMENT) -> None:
    padding = (-file_obj.tell()) % alignment
    if padding:
        file_obj.write(b"\0" * padding)


def _segment(
    *,
    name: str,
    role: str,
    shape: tuple[int, ...],
    dtype: str,
    byte_offset: int,
    byte_length: int,
    layout: str,
    source: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "source": source or name,
        "shape": list(shape),
        "dtype": dtype,
        "layout": layout,
        "byte_offset": byte_offset,
        "byte_length": byte_length,
        "byte_alignment": byte_offset & -byte_offset if byte_offset else None,
    }


def _write_bytes(file_obj, data: bytes) -> tuple[int, int]:
    _align(file_obj)
    offset = file_obj.tell()
    file_obj.write(data)
    return offset, len(data)


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
    return raw.tobytes()


def _uint8_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().to(torch.uint8).contiguous().numpy()
    return raw.tobytes()


def make_qparam(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.uint8).contiguous()
    scales_bytes = scales.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint8)
    out_features = packed.shape[0]
    return torch.cat(
        (
            packed.view(out_features, -1),
            scales_bytes.view(out_features, -1),
        ),
        dim=1,
    ).contiguous()


def make_gemm_tile_qparam(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    tile_k: int = 128,
    tile_n: int = 64,
    num_aie_columns: int = 8,
) -> torch.Tensor:
    """Pack W4A16 weights in the B-tile order needed by fused GEMM.

    The existing row-major qparam layout is good for GEMV because each core owns
    contiguous output rows. GEMM streams B by K/N tiles, so the fast path needs a
    second offline layout where each ``(tile_k, tile_n)`` weight tile is
    contiguous and carries only the scales for that K tile.
    """

    if tile_k != group_size:
        raise ValueError(
            "tile-major W4A16 GEMM currently requires tile_k == group_size "
            f"(tile_k={tile_k}, group_size={group_size})"
        )
    if in_features % tile_k != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by tile_k ({tile_k})"
        )
    if out_features % tile_n != 0:
        raise ValueError(
            f"out_features ({out_features}) must be divisible by tile_n ({tile_n})"
        )

    packed = packed.detach().cpu().to(torch.uint8).contiguous()
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    biased = torch.stack((low, high), dim=-1).flatten(1)[:, :in_features].contiguous()
    scales_bytes = scales.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint8)
    k_tiles = in_features // tile_k
    n_tiles = out_features // tile_n
    if n_tiles % num_aie_columns != 0:
        raise ValueError(
            f"out_features ({out_features}) must provide a whole number of "
            f"tile_n ({tile_n}) tiles per AIE column ({num_aie_columns})"
        )
    n_tile_groups = n_tiles // num_aie_columns
    qvalue_tile_bytes = tile_k
    scale_tile_bytes = 2
    row_bytes = ((qvalue_tile_bytes + scale_tile_bytes + 31) // 32) * 32
    tiled = torch.zeros(
        (num_aie_columns, n_tile_groups, k_tiles, tile_n, row_bytes),
        dtype=torch.uint8,
    )

    for col in range(num_aie_columns):
        for n_group in range(n_tile_groups):
            n_tile = n_group * num_aie_columns + col
            row_start = n_tile * tile_n
            row_end = row_start + tile_n
            for k_tile in range(k_tiles):
                k_start = k_tile * tile_k
                k_end = k_start + tile_k
                scale_byte_start = k_tile * scale_tile_bytes
                scale_byte_end = scale_byte_start + scale_tile_bytes
                tiled[col, n_group, k_tile, :, :qvalue_tile_bytes] = biased[
                    row_start:row_end,
                    k_start:k_end,
                ]
                tiled[
                    col,
                    n_group,
                    k_tile,
                    :,
                    qvalue_tile_bytes : qvalue_tile_bytes + scale_tile_bytes,
                ] = scales_bytes[row_start:row_end, scale_byte_start:scale_byte_end]

    return tiled.contiguous()


def make_gemm_bf16_tile(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    tile_k: int = 128,
    tile_n: int = 64,
    num_aie_columns: int = 8,
) -> torch.Tensor:
    """Pack pre-dequantized bf16 B tiles for the fused GEMM.

    The AIE kernel consumes each ``(tile_k, tile_n)`` B tile as
    ``(tile_k // 8, tile_n // 8, 8, 8)``. That is the natural ``s x t``
    subtile order for ``aie::mmul<4,8,8>`` and avoids transposing the weight
    tile at runtime.
    """

    if tile_k != group_size:
        raise ValueError(
            "bf16 tile GEMM currently requires tile_k == group_size "
            f"(tile_k={tile_k}, group_size={group_size})"
        )
    if in_features % tile_k != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by tile_k ({tile_k})"
        )
    if out_features % tile_n != 0:
        raise ValueError(
            f"out_features ({out_features}) must be divisible by tile_n ({tile_n})"
        )
    mmul_s = 8
    mmul_t = 8
    if tile_k % mmul_s != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by {mmul_s}")
    if tile_n % mmul_t != 0:
        raise ValueError(f"tile_n ({tile_n}) must be divisible by {mmul_t}")

    packed = packed.detach().cpu().to(torch.uint8).contiguous()
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    biased = torch.stack((low, high), dim=-1).flatten(1)[:, :in_features]
    signed = biased.to(torch.float32) - 8.0
    scales_f32 = scales.detach().cpu().to(torch.float32).contiguous()

    k_tiles = in_features // tile_k
    n_tiles = out_features // tile_n
    if n_tiles % num_aie_columns != 0:
        raise ValueError(
            f"out_features ({out_features}) must provide a whole number of "
            f"tile_n ({tile_n}) tiles per AIE column ({num_aie_columns})"
        )
    n_tile_groups = n_tiles // num_aie_columns
    tiled = torch.empty(
        (
            num_aie_columns,
            n_tile_groups,
            k_tiles,
            tile_k // mmul_s,
            tile_n // mmul_t,
            mmul_s,
            mmul_t,
        ),
        dtype=torch.bfloat16,
    )

    for col in range(num_aie_columns):
        for n_group in range(n_tile_groups):
            n_tile = n_group * num_aie_columns + col
            row_start = n_tile * tile_n
            row_end = row_start + tile_n
            for k_tile in range(k_tiles):
                k_start = k_tile * tile_k
                k_end = k_start + tile_k
                tile = (
                    signed[row_start:row_end, k_start:k_end]
                    * scales_f32[row_start:row_end, k_tile].view(tile_n, 1)
                )
                row_major_b = tile.t().contiguous().to(torch.bfloat16)
                tiled[col, n_group, k_tile] = (
                    row_major_b.view(
                        tile_k // mmul_s,
                        mmul_s,
                        tile_n // mmul_t,
                        mmul_t,
                    )
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

    return tiled.contiguous()


def pack_biased_int4_bytes(signed: torch.Tensor) -> torch.Tensor:
    if signed.dtype != torch.int8:
        signed = signed.to(torch.int8)
    out_features, in_features = signed.shape
    if in_features % 2:
        signed = torch.cat([signed, torch.zeros(out_features, 1, dtype=torch.int8)], dim=1)
    biased = (signed.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    low = biased[:, 0::2]
    high = torch.bitwise_left_shift(biased[:, 1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def quantize_dense_to_biased_int4(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.detach().cpu().to(torch.float32).contiguous()
    out_features, in_features = weight.shape
    num_groups = math.ceil(in_features / group_size)
    signed = torch.empty((out_features, in_features), dtype=torch.int8)
    scales = torch.empty((out_features, num_groups), dtype=torch.bfloat16)
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        group = weight[:, start:end]
        scale = group.abs().amax(dim=1).clamp(min=1e-8) / 7.0
        q = torch.round(group / scale[:, None]).clamp(-8, 7).to(torch.int8)
        signed[:, start:end] = q
        scales[:, group_idx] = scale.to(torch.bfloat16)
    return pack_biased_int4_bytes(signed), scales.contiguous()


def write_packed_inference_artifact(
    model_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else default_packed_dir(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text())
    quant_config = config.get("quantization_config") or {}
    bits = int(quant_config.get("bits", 4))
    if bits != 4:
        raise ValueError(f"only W4A16 checkpoints are supported, got bits={bits}")
    if not bool(quant_config.get("sym", True)):
        raise ValueError("only symmetric AutoGPTQ W4A16 checkpoints are supported")
    group_size = int(quant_config.get("group_size", 128))

    tensors = _load_safetensors(model_dir)
    linears = sorted(key[: -len(".qweight")] for key in tensors if key.endswith(".qweight"))
    linears_set = set(linears)

    tmp_data = output_dir / f"{PACKED_DATA}.tmp"
    data_path = output_dir / PACKED_DATA
    manifest_path = output_dir / PACKED_MANIFEST
    tmp_manifest = output_dir / f"{PACKED_MANIFEST}.tmp"

    manifest: dict[str, Any] = {
        "format": PACKED_FORMAT,
        "data_file": PACKED_DATA,
        "alignment": SEGMENT_ALIGNMENT,
        "source_model_dir": str(model_dir),
        "model_config": {
            "vocab_size": config.get("vocab_size"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "intermediate_size": config.get("intermediate_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        },
        "quantization_config": quant_config,
        "dense": {},
        "linears": {},
    }

    with open(tmp_data, "wb") as f:
        for name in sorted(tensors):
            if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
                continue
            tensor = tensors[name]
            offset, length = _write_bytes(f, _bf16_bytes(tensor))
            manifest["dense"][name] = _segment(
                name=name,
                role="dense_bf16",
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype="bfloat16",
                byte_offset=offset,
                byte_length=length,
                layout="contiguous",
            )

        for prefix in linears:
            qweight = tensors[f"{prefix}.qweight"]
            qzeros = tensors[f"{prefix}.qzeros"]
            scales = tensors[f"{prefix}.scales"]
            in_features = int(qweight.shape[0] * 8)
            out_features = int(qweight.shape[1])
            if prefix not in linears_set:
                raise AssertionError(prefix)
            signed = autogptq_to_signed_int4(
                qweight=qweight,
                qzeros=qzeros,
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
            )
            packed = pack_biased_int4_bytes(signed)
            scales_out_major = scales.t().contiguous()

            qparam = make_qparam(packed, scales_out_major)
            qparam_offset, qparam_length = _write_bytes(f, _uint8_bytes(qparam))
            linear_entry = {
                "name": prefix,
                "in_features": in_features,
                "out_features": out_features,
                "group_size": group_size,
                "num_groups": int(scales.shape[0]),
                "qparam": _segment(
                    name=f"{prefix}.qparam_biased_int4_bf16_scale",
                    role="linear_qparam_biased_int4_bf16_scale",
                    source=f"{prefix}.qweight",
                    shape=tuple(int(dim) for dim in qparam.shape),
                    dtype="uint8",
                    byte_offset=qparam_offset,
                    byte_length=qparam_length,
                    layout="out_major_qweight_then_bf16_scales",
                ),
                "zero_bias": 8,
            }
            if prefix != "lm_head":
                gemm_weight = make_gemm_bf16_tile(
                    packed,
                    scales_out_major,
                    in_features=in_features,
                    out_features=out_features,
                    group_size=group_size,
                )
                gemm_weight_offset, gemm_weight_length = _write_bytes(
                    f,
                    _bf16_bytes(gemm_weight),
                )
                linear_entry["gemm_weight"] = _segment(
                    name=f"{prefix}.gemm_tile_bf16_weight",
                    role="linear_gemm_tile_bf16_weight",
                    source=f"{prefix}.qweight",
                    shape=tuple(int(dim) for dim in gemm_weight.shape),
                    dtype="bfloat16",
                    byte_offset=gemm_weight_offset,
                    byte_length=gemm_weight_length,
                    layout="col_major_n_group_k_tile_kblock_nblock_s_t_bf16_mmul",
                )
                linear_entry["gemm_tile"] = {
                    "tile_k": 128,
                    "tile_n": 64,
                    "num_aie_columns": 8,
                }
            manifest["linears"][prefix] = linear_entry
        if "lm_head" not in manifest["linears"]:
            lm_source = "lm_head.weight" if "lm_head.weight" in tensors else "model.embed_tokens.weight"
            if lm_source in tensors:
                packed, scales_out_major = quantize_dense_to_biased_int4(
                    tensors[lm_source],
                    group_size=group_size,
                )
                out_features, _packed_k = packed.shape
                in_features = int(tensors[lm_source].shape[1])
                qparam = make_qparam(packed, scales_out_major)
                qparam_offset, qparam_length = _write_bytes(f, _uint8_bytes(qparam))
                lm_head_entry = {
                    "name": "lm_head",
                    "in_features": in_features,
                    "out_features": int(out_features),
                    "group_size": group_size,
                    "num_groups": int(scales_out_major.shape[1]),
                    "qparam": _segment(
                        name="lm_head.qparam_biased_int4_bf16_scale",
                        role="linear_qparam_biased_int4_bf16_scale",
                        source=lm_source,
                        shape=tuple(int(dim) for dim in qparam.shape),
                        dtype="uint8",
                        byte_offset=qparam_offset,
                        byte_length=qparam_length,
                        layout="out_major_qweight_then_bf16_scales",
                    ),
                    "zero_bias": 8,
                    "synthetic": True,
                }
                manifest["linears"]["lm_head"] = lm_head_entry
        _align(f)
        manifest["total_bytes"] = f.tell()

    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_data.replace(data_path)
    tmp_manifest.replace(manifest_path)
    return manifest


class PackedInferenceStore:
    def __init__(self, packed_dir: str | Path) -> None:
        packed_dir = Path(packed_dir).expanduser().resolve()
        manifest_path = packed_dir / PACKED_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing packed manifest: {manifest_path}")
        self.packed_dir = packed_dir
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != PACKED_FORMAT:
            raise RuntimeError(
                f"unsupported packed format {self.manifest.get('format')!r}; "
                f"expected {PACKED_FORMAT!r}"
            )
        self.data_path = packed_dir / self.manifest["data_file"]
        self._data = np.memmap(self.data_path, dtype=np.uint8, mode="r+")

    def _numpy_view(self, segment: dict[str, Any], dtype: np.dtype) -> np.ndarray:
        return np.ndarray(
            shape=tuple(int(dim) for dim in segment["shape"]),
            dtype=dtype,
            buffer=self._data,
            offset=int(segment["byte_offset"]),
        )

    def dense(self, name: str) -> torch.Tensor:
        try:
            segment = self.manifest["dense"][name]
        except KeyError as exc:
            raise KeyError(f"missing dense tensor {name!r} in packed store") from exc
        raw = self._numpy_view(segment, np.dtype(np.uint16))
        return torch.from_numpy(raw).view(torch.bfloat16)

    def linear_qparam(self, prefix: str) -> tuple[dict[str, Any], torch.Tensor]:
        try:
            spec = self.manifest["linears"][prefix]
        except KeyError as exc:
            raise KeyError(f"missing packed linear {prefix!r}") from exc
        return spec, torch.from_numpy(self._numpy_view(spec["qparam"], np.dtype(np.uint8)))

    def linear_gemm_weight(self, prefix: str) -> tuple[dict[str, Any], torch.Tensor]:
        try:
            spec = self.manifest["linears"][prefix]
        except KeyError as exc:
            raise KeyError(f"missing packed linear {prefix!r}") from exc
        if "gemm_weight" not in spec:
            raise KeyError(
                f"packed linear {prefix!r} does not contain GEMM bf16 tile weight; "
                "rerun `python -m models.quantized_qwen3.pack`"
            )
        return spec, torch.from_numpy(
            self._numpy_view(spec["gemm_weight"], np.dtype(np.uint16))
        ).view(torch.bfloat16)

    def linear_gemm_qparam(self, prefix: str) -> tuple[dict[str, Any], torch.Tensor]:
        try:
            spec = self.manifest["linears"][prefix]
        except KeyError as exc:
            raise KeyError(f"missing packed linear {prefix!r}") from exc
        if "gemm_qparam" not in spec:
            raise KeyError(
                f"packed linear {prefix!r} does not contain legacy GEMM tile qparam; "
                "rerun `python -m models.quantized_qwen3.pack`"
            )
        return spec, torch.from_numpy(
            self._numpy_view(spec["gemm_qparam"], np.dtype(np.uint8))
        )

    def linear_segments(self, prefix: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        spec, _qparam = self.linear_qparam(prefix)
        in_features = int(spec["in_features"])
        out_features = int(spec["out_features"])
        num_groups = int(spec["num_groups"])
        packed_k = (in_features + 1) // 2
        qparam_segment = spec["qparam"]
        qparam_row_bytes = int(qparam_segment["shape"][1])
        qparam_offset = int(qparam_segment["byte_offset"])
        packed_np = np.ndarray(
            shape=(out_features, packed_k),
            dtype=np.uint8,
            buffer=self._data,
            offset=qparam_offset,
            strides=(qparam_row_bytes, 1),
        )
        scales_raw = np.ndarray(
            shape=(out_features, num_groups),
            dtype=np.uint16,
            buffer=self._data,
            offset=qparam_offset + packed_k,
            strides=(qparam_row_bytes, np.dtype(np.uint16).itemsize),
        )
        packed = torch.from_numpy(packed_np)
        scales = torch.from_numpy(scales_raw).view(torch.bfloat16)
        return spec, packed, scales

    def has_linear(self, prefix: str) -> bool:
        return prefix in self.manifest.get("linears", {})
