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
class LlamaChunkedPrefillAttention(MLIROperator):
    """Chunked Llama prefill attention over decode-style packet caches.

    This is the prefill counterpart of ``LlamaChunkedAttention``.  It keeps the
    same packed KV cache format for past tokens, but handles multiple query
    tokens in one dispatch by taking the current chunk K/V as separate inputs.
    The kernel applies the local causal rule for the current chunk, so callers
    do not need to materialize a full [seq, seq] mask or attention matrix.
    """

    max_seq_len: int
    query_len: int
    num_kv_groups: int = 8
    q_heads_per_group: int = 4
    q_head_block_size: int = 2
    head_dim: int = 64
    chunk_size: int = 64
    kernel_vector_size: int = field(default=32, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "max_seq_len": "msl",
        "query_len": "qlen",
        "num_kv_groups": "kvg",
        "q_heads_per_group": "qhpg",
        "q_head_block_size": "qhblk",
        "head_dim": "hd",
        "chunk_size": "chunk",
    }

    def __post_init__(self):
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.query_len <= 0:
            raise ValueError("query_len must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_seq_len % self.chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        if self.num_kv_groups <= 0:
            raise ValueError("num_kv_groups must be positive")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.q_head_block_size <= 0:
            raise ValueError("q_head_block_size must be positive")
        if self.q_heads_per_group % self.q_head_block_size != 0:
            raise ValueError("q_heads_per_group must be divisible by q_head_block_size")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % self.kernel_vector_size != 0:
            raise ValueError("head_dim must be divisible by kernel_vector_size")
        MLIROperator.__init__(self, context=self.context)

    @property
    def num_chunks(self):
        return self.max_seq_len // self.chunk_size

    @property
    def q_heads(self):
        return self.num_kv_groups * self.q_heads_per_group

    @property
    def q_head_blocks_per_group(self):
        return self.q_heads_per_group // self.q_head_block_size

    @property
    def logical_groups(self):
        return self.num_kv_groups * self.q_head_blocks_per_group

    @property
    def q_elements_per_group(self):
        return self.query_len * self.q_head_block_size * self.head_dim

    @property
    def current_kv_elements_per_group(self):
        return self.query_len * self.head_dim

    @property
    def q_current_elements_per_group(self):
        return self.q_elements_per_group + 2 * self.current_kv_elements_per_group

    @property
    def q_elements(self):
        return self.logical_groups * self.q_elements_per_group

    @property
    def q_current_elements(self):
        return self.logical_groups * self.q_current_elements_per_group

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
            "llama_chunked_prefill_attention_"
            f"qlen{self.query_len}_qhblk{self.q_head_block_size}_"
            f"hd{self.head_dim}_chunk{self.chunk_size}_"
            f"kv{self.kernel_vector_size}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "llama_chunked_prefill_attention",
                (
                    aie_utils.get_current_device(),
                    self.max_seq_len,
                    self.query_len,
                    self.num_kv_groups,
                    self.q_heads_per_group,
                    self.q_head_block_size,
                    self.head_dim,
                    self.chunk_size,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name,
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
                    f"-DLLAMA_PREFILL_QUERY_LEN={self.query_len}",
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
                "in",
                (
                    self.num_kv_groups,
                    self.q_head_blocks_per_group,
                    self.q_current_elements_per_group,
                ),
            ),
            AIERuntimeArgSpec("in", (self.packed_elements,)),
            AIERuntimeArgSpec(
                "out",
                (
                    self.num_kv_groups,
                    self.q_head_blocks_per_group,
                    self.query_len,
                    self.q_head_block_size,
                    self.head_dim,
                ),
            ),
        ]
