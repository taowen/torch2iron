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

from models.fast_qwen3.kv_plane_reference import kv_plane_total_elements


@dataclass
class QwenPlaneAttentionCurrent(MLIROperator):
    """Grouped decode attention over the FastFlowLM-style four-plane KV cache."""

    packet_seq_len: int
    attend_seq_len: int
    current_slot: int
    q_heads_per_group: int = 2
    head_dim: int = 128
    tile_size: int = 16
    plane_fifo_depth: int = 2
    kernel_vector_size: int = field(default=32, repr=False)
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "packet_seq_len": "psl",
        "attend_seq_len": "asl",
        "current_slot": "slot",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
        "tile_size": "tile",
        "plane_fifo_depth": "pfdepth",
    }

    def __post_init__(self) -> None:
        if self.packet_seq_len <= 0:
            raise ValueError("packet_seq_len must be positive")
        if self.attend_seq_len <= 0 or self.attend_seq_len > self.packet_seq_len:
            raise ValueError("attend_seq_len must be in (0, packet_seq_len]")
        if self.rounded_attend_seq_len > self.packet_seq_len:
            raise ValueError("rounded attend_seq_len must fit inside packet_seq_len")
        if self.current_slot < 0 or self.current_slot >= self.attend_seq_len:
            raise ValueError("current_slot must be inside attend_seq_len")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % self.kernel_vector_size != 0:
            raise ValueError("head_dim must be divisible by kernel_vector_size")
        if self.plane_fifo_depth <= 0:
            raise ValueError("plane_fifo_depth must be positive")
        MLIROperator.__init__(self, context=self.context)

    @property
    def num_kv_groups(self) -> int:
        return 8

    @property
    def num_chunks(self) -> int:
        return (self.attend_seq_len + self.tile_size - 1) // self.tile_size

    @property
    def rounded_attend_seq_len(self) -> int:
        return self.num_chunks * self.tile_size

    @property
    def q_elements_per_group(self) -> int:
        return self.q_heads_per_group * self.head_dim

    @property
    def q_current_elements_per_group(self) -> int:
        return self.q_elements_per_group + 2 * self.head_dim

    @property
    def q_current_elements(self) -> int:
        return self.num_kv_groups * self.q_current_elements_per_group

    @property
    def q_elements(self) -> int:
        return self.num_kv_groups * self.q_elements_per_group

    @property
    def kv_plane_elements(self) -> int:
        return kv_plane_total_elements(self.packet_seq_len, self.head_dim)

    @property
    def plane_pair_chunk_elements(self) -> int:
        return 2 * self.tile_size * 4 * self.head_dim

    def _kernel_object_name(self) -> str:
        return (
            "qwen_plane_attention_current_"
            f"hd{self.head_dim}_tile{self.tile_size}_"
            f"qh{self.q_heads_per_group}_kv{self.kernel_vector_size}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen_plane_attention_current",
                (
                    aie_utils.get_current_device(),
                    self.packet_seq_len,
                    self.attend_seq_len,
                    self.current_slot,
                    self.q_heads_per_group,
                    self.head_dim,
                    self.tile_size,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                    "plane_fifo_depth": self.plane_fifo_depth,
                },
            ),
        )

    def get_kernel_artifacts(self):
        scale = 1.0 / math.sqrt(self.head_dim)
        source = self.operator_dir.parent / "qwen_chunked_attention_current" / "kernel.cc"
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                dependencies=[SourceArtifact(source)],
                extra_flags=[
                    f"-DLLAMA_HEAD_DIM={self.head_dim}",
                    f"-DLLAMA_CHUNK_SIZE={self.tile_size}",
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
            AIERuntimeArgSpec("in", (self.kv_plane_elements,), dtype=bfloat16),
            AIERuntimeArgSpec(
                "out",
                (self.num_kv_groups, self.q_heads_per_group, self.head_dim),
                dtype=bfloat16,
            ),
        ]
