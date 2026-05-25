#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import math
from typing import ClassVar

import aie.utils as aie_utils
from ml_dtypes import bfloat16

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext

from models.fast_qwen3.attention_reference import (
    decode_packet_chunk_elements,
    decode_packet_elements,
    decode_packet_elements_per_group,
)


@dataclass
class QwenChunkedAttentionCurrent(MLIROperator):
    """Append the current K/V slot and run grouped chunked decode attention.

    The worker consumes Q and the current K/V as one stream, then scans packet
    chunks.  For the current slot it bypasses the packet row and attends to the
    current K/V directly.  Packet persistence is kept as a separate small-write
    step until the fused layer path has manual placement for the update stream.
    """

    packet_seq_len: int
    attend_seq_len: int
    current_slot: int
    num_kv_groups: int = 8
    q_heads_per_group: int = 2
    head_dim: int = 128
    chunk_size: int = 64
    packed_fifo_depth: int | None = None
    kernel_vector_size: int = field(default=32, repr=False)
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "packet_seq_len": "psl",
        "attend_seq_len": "asl",
        "current_slot": "slot",
        "num_kv_groups": "kvg",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
        "chunk_size": "chunk",
        "packed_fifo_depth": "pfdepth",
    }

    def __post_init__(self) -> None:
        if self.packet_seq_len <= 0:
            raise ValueError("packet_seq_len must be positive")
        if self.attend_seq_len <= 0 or self.attend_seq_len > self.packet_seq_len:
            raise ValueError("attend_seq_len must be in (0, packet_seq_len]")
        if self.packet_seq_len % self.chunk_size != 0:
            raise ValueError("packet_seq_len must be divisible by chunk_size")
        if self.attend_seq_len % self.chunk_size != 0:
            raise ValueError("attend_seq_len must be divisible by chunk_size")
        if self.current_slot < 0 or self.current_slot >= self.attend_seq_len:
            raise ValueError("current_slot must be inside attend_seq_len")
        if self.num_kv_groups <= 0:
            raise ValueError("num_kv_groups must be positive")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % self.kernel_vector_size != 0:
            raise ValueError("head_dim must be divisible by kernel_vector_size")

        packed_object_bytes = self.packed_chunk_elements * 2
        if self.packed_fifo_depth is None:
            self.packed_fifo_depth = 2 if packed_object_bytes <= 24 * 1024 else 1
        if self.packed_fifo_depth <= 0:
            raise ValueError("packed_fifo_depth must be positive")
        if self.packed_fifo_depth > 1 and packed_object_bytes > 24 * 1024:
            raise ValueError(
                "packed_fifo_depth > 1 requires smaller chunks; "
                f"one packed chunk is {packed_object_bytes} bytes"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def num_chunks(self) -> int:
        return self.attend_seq_len // self.chunk_size

    @property
    def q_elements_per_group(self) -> int:
        return self.q_heads_per_group * self.head_dim

    @property
    def q_elements(self) -> int:
        return self.num_kv_groups * self.q_elements_per_group

    @property
    def q_current_elements_per_group(self) -> int:
        return self.q_elements_per_group + 2 * self.head_dim

    @property
    def q_current_elements(self) -> int:
        return self.num_kv_groups * self.q_current_elements_per_group

    @property
    def packed_chunk_elements(self) -> int:
        return decode_packet_chunk_elements(self.chunk_size, self.head_dim)

    @property
    def packet_elements_per_group(self) -> int:
        return decode_packet_elements_per_group(
            self.packet_seq_len,
            self.chunk_size,
            self.head_dim,
        )

    @property
    def active_packet_elements_per_group(self) -> int:
        return self.num_chunks * self.packed_chunk_elements

    @property
    def packet_elements(self) -> int:
        return decode_packet_elements(
            self.num_kv_groups,
            self.packet_seq_len,
            self.chunk_size,
            self.head_dim,
        )

    def _kernel_object_name(self) -> str:
        return (
            "qwen_chunked_attention_current_"
            f"hd{self.head_dim}_chunk{self.chunk_size}_"
            f"qh{self.q_heads_per_group}_kv{self.kernel_vector_size}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen_chunked_attention_current",
                (
                    aie_utils.get_current_device(),
                    self.packet_seq_len,
                    self.attend_seq_len,
                    self.current_slot,
                    self.num_kv_groups,
                    self.q_heads_per_group,
                    self.head_dim,
                    self.chunk_size,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                    "packed_fifo_depth": self.packed_fifo_depth,
                },
            ),
        )

    def get_kernel_artifacts(self):
        scale = 1.0 / math.sqrt(self.head_dim)
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[
                    f"-DLLAMA_HEAD_DIM={self.head_dim}",
                    f"-DLLAMA_CHUNK_SIZE={self.chunk_size}",
                    f"-DLLAMA_Q_HEADS_PER_GROUP={self.q_heads_per_group}",
                    f"-DLLAMA_ATTN_SCALE={scale:.17g}f",
                    f"-DLLAMA_VEC_SIZE={self.kernel_vector_size}",
                ],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec(
                "in",
                (self.num_kv_groups, self.q_current_elements_per_group),
                dtype=bfloat16,
            ),
            AIERuntimeArgSpec("in", (self.packet_elements,), dtype=bfloat16),
            AIERuntimeArgSpec(
                "out",
                (self.num_kv_groups, self.q_heads_per_group, self.head_dim),
                dtype=bfloat16,
            ),
        ]
