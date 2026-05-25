#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q4NX-style group32 patch layout for fast Qwen3 decode layers.

This layout follows the FastFlowLM-facing facts we can validate from host-side
control sequences:

* one chunk covers 32 output rows x 256 input columns;
* each chunk stores 256 bf16 scales, 256 bf16 zero-points, then 4096 bytes of
  packed uint4 values;
* dequantization is ``weight = (q4 - zero_point) * scale``;
* a host-visible DDR patch covers 64 output rows x full K, with K chunks inside.

The code here is intentionally CPU/reference oriented.  It fixes the artifact
contract that the AIE fused projection kernels will consume.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


Q4NX_GROUP_SIZE = 32
Q4NX_OUT_CHUNK = 32
Q4NX_IN_CHUNK = 256
Q4NX_PATCH_OUT_ROWS = 64
Q4NX_SCALE_COUNT = Q4NX_OUT_CHUNK * (Q4NX_IN_CHUNK // Q4NX_GROUP_SIZE)
Q4NX_SCALE_BYTES = Q4NX_SCALE_COUNT * 2
Q4NX_ZERO_BYTES = Q4NX_SCALE_COUNT * 2
Q4NX_INT4_BYTES = Q4NX_OUT_CHUNK * Q4NX_IN_CHUNK // 2
Q4NX_CHUNK_BYTES = Q4NX_SCALE_BYTES + Q4NX_ZERO_BYTES + Q4NX_INT4_BYTES


@dataclass(frozen=True)
class Q4NXLinearSpec:
    name: str
    in_features: int
    out_features: int
    padded_out_features: int
    patch_out_rows: int
    in_chunk: int
    out_chunk: int
    group_size: int
    patch_bytes: int
    byte_offset: int
    byte_length: int


def padded_out_features(out_features: int) -> int:
    return (
        (out_features + Q4NX_PATCH_OUT_ROWS - 1)
        // Q4NX_PATCH_OUT_ROWS
        * Q4NX_PATCH_OUT_ROWS
    )


def linear_patch_bytes(in_features: int) -> int:
    _validate_in_features(in_features)
    return (in_features // Q4NX_IN_CHUNK) * 2 * Q4NX_CHUNK_BYTES


def linear_total_bytes(in_features: int, out_features: int) -> int:
    return padded_out_features(out_features) // Q4NX_PATCH_OUT_ROWS * linear_patch_bytes(
        in_features
    )


def _validate_in_features(in_features: int) -> None:
    if in_features % Q4NX_IN_CHUNK != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by {Q4NX_IN_CHUNK}"
        )


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16)
    return raw.numpy().tobytes()


def _bf16_from_bytes(data: bytes | memoryview, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.frombuffer(data, dtype=np.uint16).reshape(shape)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)


def _pack_uint4_rows(qvalues: torch.Tensor) -> bytes:
    qvalues = qvalues.detach().cpu().to(torch.uint8).contiguous()
    if tuple(qvalues.shape) != (Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK):
        raise ValueError(
            "qvalues must have shape "
            f"({Q4NX_OUT_CHUNK}, {Q4NX_IN_CHUNK}), got {tuple(qvalues.shape)}"
        )
    low = torch.bitwise_and(qvalues[:, 0::2], 0x0F)
    high = torch.bitwise_left_shift(torch.bitwise_and(qvalues[:, 1::2], 0x0F), 4)
    return torch.bitwise_or(low, high).contiguous().numpy().tobytes()


def _unpack_uint4_rows(data: bytes | memoryview) -> torch.Tensor:
    raw = torch.from_numpy(
        np.frombuffer(data, dtype=np.uint8)
        .copy()
        .reshape(Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK // 2)
    )
    low = torch.bitwise_and(raw, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(raw, 4), 0x0F)
    return torch.stack((low, high), dim=-1).flatten(1).contiguous()


def quantize_chunk(chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(chunk.shape) != (Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK):
        raise ValueError(
            "chunk must have shape "
            f"({Q4NX_OUT_CHUNK}, {Q4NX_IN_CHUNK}), got {tuple(chunk.shape)}"
        )
    chunk = chunk.detach().cpu().to(torch.float32).contiguous()
    groups = chunk.view(Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK // Q4NX_GROUP_SIZE, Q4NX_GROUP_SIZE)
    min_values = groups.amin(dim=2)
    max_values = groups.amax(dim=2)
    scale = ((max_values - min_values) / 15.0).clamp(min=1e-8)
    zero = (-min_values / scale).clamp(0.0, 15.0)
    q = torch.round(groups / scale.unsqueeze(-1) + zero.unsqueeze(-1))
    q = q.clamp(0, 15).to(torch.uint8).view(Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK)
    return q, scale.to(torch.bfloat16).contiguous(), zero.to(torch.bfloat16).contiguous()


def dequantize_chunk(
    qvalues: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    groups = qvalues.to(torch.float32).view(
        Q4NX_OUT_CHUNK,
        Q4NX_IN_CHUNK // Q4NX_GROUP_SIZE,
        Q4NX_GROUP_SIZE,
    )
    return (
        (groups - zero.to(torch.float32).unsqueeze(-1))
        * scale.to(torch.float32).unsqueeze(-1)
    ).view(Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK)


def pack_chunk(chunk: torch.Tensor) -> bytes:
    qvalues, scale, zero = quantize_chunk(chunk)
    return b"".join(
        (
            _bf16_bytes(scale),
            _bf16_bytes(zero),
            _pack_uint4_rows(qvalues),
        )
    )


def unpack_chunk(data: bytes | memoryview) -> torch.Tensor:
    if len(data) != Q4NX_CHUNK_BYTES:
        raise ValueError(f"Q4NX chunk must be {Q4NX_CHUNK_BYTES} bytes, got {len(data)}")
    scale_begin = 0
    zero_begin = Q4NX_SCALE_BYTES
    int4_begin = Q4NX_SCALE_BYTES + Q4NX_ZERO_BYTES
    scale = _bf16_from_bytes(
        memoryview(data)[scale_begin:zero_begin],
        (Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK // Q4NX_GROUP_SIZE),
    )
    zero = _bf16_from_bytes(
        memoryview(data)[zero_begin:int4_begin],
        (Q4NX_OUT_CHUNK, Q4NX_IN_CHUNK // Q4NX_GROUP_SIZE),
    )
    qvalues = _unpack_uint4_rows(memoryview(data)[int4_begin:])
    return dequantize_chunk(qvalues, scale, zero)


def pack_linear_weight(weight: torch.Tensor) -> tuple[bytes, Q4NXLinearSpec]:
    weight = weight.detach().cpu().to(torch.float32).contiguous()
    if weight.dim() != 2:
        raise ValueError(f"linear weight must be rank-2, got shape {tuple(weight.shape)}")
    out_features, in_features = (int(weight.shape[0]), int(weight.shape[1]))
    _validate_in_features(in_features)

    padded_out = padded_out_features(out_features)
    if padded_out != out_features:
        pad = torch.zeros((padded_out - out_features, in_features), dtype=torch.float32)
        weight = torch.cat((weight, pad), dim=0)

    blocks: list[bytes] = []
    for out_start in range(0, padded_out, Q4NX_PATCH_OUT_ROWS):
        out_block = weight[out_start : out_start + Q4NX_PATCH_OUT_ROWS, :]
        for in_start in range(0, in_features, Q4NX_IN_CHUNK):
            for out_half in (0, Q4NX_OUT_CHUNK):
                chunk = out_block[
                    out_half : out_half + Q4NX_OUT_CHUNK,
                    in_start : in_start + Q4NX_IN_CHUNK,
                ]
                blocks.append(pack_chunk(chunk))

    data = b"".join(blocks)
    spec = Q4NXLinearSpec(
        name="",
        in_features=in_features,
        out_features=out_features,
        padded_out_features=padded_out,
        patch_out_rows=Q4NX_PATCH_OUT_ROWS,
        in_chunk=Q4NX_IN_CHUNK,
        out_chunk=Q4NX_OUT_CHUNK,
        group_size=Q4NX_GROUP_SIZE,
        patch_bytes=linear_patch_bytes(in_features),
        byte_offset=0,
        byte_length=len(data),
    )
    return data, spec


def dequantize_linear(data: bytes | memoryview, spec: Q4NXLinearSpec) -> torch.Tensor:
    expected_bytes = linear_total_bytes(spec.in_features, spec.out_features)
    if len(data) != expected_bytes:
        raise ValueError(f"linear data must be {expected_bytes} bytes, got {len(data)}")

    weight = torch.empty(
        (spec.padded_out_features, spec.in_features),
        dtype=torch.float32,
    )
    offset = 0
    for out_start in range(0, spec.padded_out_features, Q4NX_PATCH_OUT_ROWS):
        for in_start in range(0, spec.in_features, Q4NX_IN_CHUNK):
            for out_half in (0, Q4NX_OUT_CHUNK):
                chunk = unpack_chunk(memoryview(data)[offset : offset + Q4NX_CHUNK_BYTES])
                offset += Q4NX_CHUNK_BYTES
                weight[
                    out_start + out_half : out_start + out_half + Q4NX_OUT_CHUNK,
                    in_start : in_start + Q4NX_IN_CHUNK,
                ] = chunk
    return weight[: spec.out_features, :]


def _q4nx_single_k_patch_reference(
    x: torch.Tensor,
    patch: bytes | memoryview,
) -> torch.Tensor:
    if x.shape[-1] != Q4NX_IN_CHUNK:
        raise ValueError(
            f"input last dim {x.shape[-1]} does not match Q4NX patch K={Q4NX_IN_CHUNK}"
        )
    if len(patch) != 2 * Q4NX_CHUNK_BYTES:
        raise ValueError(
            f"Q4NX output patch must be {2 * Q4NX_CHUNK_BYTES} bytes, got {len(patch)}"
        )
    first = unpack_chunk(memoryview(patch)[:Q4NX_CHUNK_BYTES])
    second = unpack_chunk(memoryview(patch)[Q4NX_CHUNK_BYTES:])
    weight = torch.cat((first, second), dim=0)
    out = x.to(torch.float32).reshape(-1, Q4NX_IN_CHUNK).matmul(weight.t())
    return out.reshape(*x.shape[:-1], Q4NX_PATCH_OUT_ROWS).to(torch.bfloat16)


def q4nx_output_patch_reference(
    x: torch.Tensor,
    patch: bytes | memoryview,
    *,
    bf16_partial_accum: bool = False,
) -> torch.Tensor:
    in_features = int(x.shape[-1])
    _validate_in_features(in_features)
    expected_bytes = linear_patch_bytes(in_features)
    if len(patch) != expected_bytes:
        raise ValueError(
            f"Q4NX output patch must be {expected_bytes} bytes, got {len(patch)}"
        )
    output = torch.zeros(
        (*x.shape[:-1], Q4NX_PATCH_OUT_ROWS),
        dtype=torch.float32,
    )
    chunk_bytes = 2 * Q4NX_CHUNK_BYTES
    for k_start in range(0, in_features, Q4NX_IN_CHUNK):
        k_idx = k_start // Q4NX_IN_CHUNK
        contribution = _q4nx_single_k_patch_reference(
            x[..., k_start : k_start + Q4NX_IN_CHUNK],
            memoryview(patch)[k_idx * chunk_bytes : (k_idx + 1) * chunk_bytes],
        )
        output = output + contribution.to(torch.float32)
        if bf16_partial_accum:
            output = output.to(torch.bfloat16).to(torch.float32)
    return output.to(torch.bfloat16)


def q4nx_patch_reference(x: torch.Tensor, patch: bytes | memoryview) -> torch.Tensor:
    return q4nx_output_patch_reference(x, patch)


def q4nx_linear_reference(
    x: torch.Tensor,
    data: bytes | memoryview,
    spec: Q4NXLinearSpec,
) -> torch.Tensor:
    if x.shape[-1] != spec.in_features:
        raise ValueError(
            f"input last dim {x.shape[-1]} does not match {spec.in_features}"
        )
    weight = dequantize_linear(data, spec)
    out = x.to(torch.float32).reshape(-1, spec.in_features).matmul(weight.t())
    return out.reshape(*x.shape[:-1], spec.out_features).to(torch.bfloat16)


def spec_from_manifest(name: str, entry) -> Q4NXLinearSpec:
    return Q4NXLinearSpec(
        name=name,
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        padded_out_features=int(entry["padded_out_features"]),
        patch_out_rows=int(entry["patch_out_rows"]),
        in_chunk=int(entry["in_chunk"]),
        out_chunk=int(entry["out_chunk"]),
        group_size=int(entry["group_size"]),
        patch_bytes=int(entry["patch_bytes"]),
        byte_offset=int(entry["byte_offset"]),
        byte_length=int(entry["byte_length"]),
    )


def manifest_entry(spec: Q4NXLinearSpec, *, name: str, byte_offset: int) -> dict:
    return {
        "name": name,
        "layout": "q4nx_group32_patch_out64_kmajor_scale_zero_int4",
        "in_features": spec.in_features,
        "out_features": spec.out_features,
        "padded_out_features": spec.padded_out_features,
        "patch_out_rows": spec.patch_out_rows,
        "in_chunk": spec.in_chunk,
        "out_chunk": spec.out_chunk,
        "group_size": spec.group_size,
        "patch_bytes": spec.patch_bytes,
        "byte_offset": byte_offset,
        "byte_length": spec.byte_length,
    }


def num_q4nx_chunks(in_features: int, out_features: int) -> int:
    return (
        padded_out_features(out_features)
        // Q4NX_OUT_CHUNK
        * (in_features // Q4NX_IN_CHUNK)
    )


def compression_ratio(in_features: int, out_features: int) -> float:
    bf16_bytes = in_features * out_features * 2
    q4nx_bytes = linear_total_bytes(in_features, out_features)
    return bf16_bytes / q4nx_bytes


def assert_layout_constants() -> None:
    if Q4NX_CHUNK_BYTES != 5120:
        raise AssertionError(f"unexpected Q4NX chunk bytes: {Q4NX_CHUNK_BYTES}")
    if Q4NX_SCALE_COUNT != 256:
        raise AssertionError(f"unexpected Q4NX scale count: {Q4NX_SCALE_COUNT}")
    if Q4NX_INT4_BYTES != 4096:
        raise AssertionError(f"unexpected Q4NX int4 bytes: {Q4NX_INT4_BYTES}")
    if math.gcd(Q4NX_PATCH_OUT_ROWS, Q4NX_OUT_CHUNK) != Q4NX_OUT_CHUNK:
        raise AssertionError("patch rows must contain whole Q4NX output chunks")
