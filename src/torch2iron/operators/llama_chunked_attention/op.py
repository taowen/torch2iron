# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import ClassVar

import aie.utils as aie_utils

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)


@dataclass
class LlamaChunkedAttention(MLIROperator):
    """Llama decode attention specialized for fixed chunked KV packets.

    The operator consumes grouped-query attention in Llama 3.2 1B layout:

    - Q:      [num_kv_groups, q_heads_per_group, head_dim]
    - packet: [num_kv_groups, num_chunks, K_chunk + V_chunk + mask_chunk]
    - output: same layout as Q

    Each worker owns one KV group and updates all local Q heads while streaming
    each KV chunk once. This avoids physical GQA repeat and avoids materializing
    full attention score/weight tensors.
    """

    max_seq_len: int
    num_kv_groups: int = 8
    q_heads_per_group: int = 4
    head_dim: int = 64
    chunk_size: int = 64
    packed_fifo_depth: int | None = None
    kernel_vector_size: int = field(default=32, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "max_seq_len": "msl",
        "num_kv_groups": "kvg",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
        "chunk_size": "chunk",
        "packed_fifo_depth": "pfdepth",
    }

    def __post_init__(self):
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_seq_len % self.chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")
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
                "packed_fifo_depth > 1 requires smaller attention chunks; "
                f"one packed chunk is {packed_object_bytes} bytes"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def num_chunks(self):
        return self.max_seq_len // self.chunk_size

    @property
    def q_heads(self):
        return self.num_kv_groups * self.q_heads_per_group

    @property
    def q_elements(self):
        return self.q_heads * self.head_dim

    @property
    def packed_chunk_elements(self):
        return 2 * self.chunk_size * self.head_dim + self.chunk_size

    @property
    def packed_elements_per_group(self):
        return self.num_chunks * self.packed_chunk_elements

    @property
    def packed_elements(self):
        return self.num_kv_groups * self.packed_elements_per_group

    @property
    def output_elements(self):
        return self.q_elements

    @property
    def _kernel_object_name(self):
        return (
            "llama_chunked_attention_"
            f"hd{self.head_dim}_chunk{self.chunk_size}_"
            f"kv{self.kernel_vector_size}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "llama_chunked_attention",
                (
                    aie_utils.get_current_device(),
                    self.max_seq_len,
                    self.num_kv_groups,
                    self.q_heads_per_group,
                    self.head_dim,
                    self.chunk_size,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name,
                    "packed_fifo_depth": self.packed_fifo_depth,
                },
            ),
        )

    def get_kernel_artifacts(self):
        scale = 1.0 / math.sqrt(self.head_dim)
        return [
            KernelObjectArtifact(
                self._kernel_object_name,
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[
                    f"-DLLAMA_HEAD_DIM={self.head_dim}",
                    f"-DLLAMA_CHUNK_SIZE={self.chunk_size}",
                    f"-DLLAMA_ATTN_SCALE={scale:.17g}f",
                    f"-DLLAMA_VEC_SIZE={self.kernel_vector_size}",
                ],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec(
                "in", (self.num_kv_groups, self.q_heads_per_group, self.head_dim)
            ),
            AIERuntimeArgSpec("in", (self.packed_elements,)),
            AIERuntimeArgSpec(
                "out", (self.num_kv_groups, self.q_heads_per_group, self.head_dim)
            ),
        ]
