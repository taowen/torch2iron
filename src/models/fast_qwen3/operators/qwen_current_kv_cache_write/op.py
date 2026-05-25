#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

import aie.utils as aie_utils
from ml_dtypes import bfloat16

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
)
from iron.common.context import AIEContext

from models.fast_qwen3.attention_reference import (
    decode_packet_elements,
    decode_packet_elements_per_group,
)


@dataclass
class QwenCurrentKVCacheWrite(MLIROperator):
    """Write the current K/V row from q_current into packet cache.

    ``q_current`` is grouped as ``[Q heads, current K, current V]`` per KV group.
    This writer persists only the current slot.  Attention can still consume
    current K/V directly, keeping the attention worker at two input streams.
    """

    packet_seq_len: int
    current_slot: int
    num_kv_groups: int = 8
    q_heads_per_group: int = 2
    head_dim: int = 128
    chunk_size: int = 64
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "packet_seq_len": "psl",
        "current_slot": "slot",
        "num_kv_groups": "kvg",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
        "chunk_size": "chunk",
    }

    def __post_init__(self) -> None:
        if self.packet_seq_len <= 0:
            raise ValueError("packet_seq_len must be positive")
        if self.packet_seq_len % self.chunk_size != 0:
            raise ValueError("packet_seq_len must be divisible by chunk_size")
        if self.current_slot < 0 or self.current_slot >= self.packet_seq_len:
            raise ValueError("current_slot must be inside packet_seq_len")
        if self.num_kv_groups <= 0:
            raise ValueError("num_kv_groups must be positive")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        MLIROperator.__init__(self, context=self.context)

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
    def packet_elements_per_group(self) -> int:
        return decode_packet_elements_per_group(
            self.packet_seq_len,
            self.chunk_size,
            self.head_dim,
        )

    @property
    def packet_elements(self) -> int:
        return decode_packet_elements(
            self.num_kv_groups,
            self.packet_seq_len,
            self.chunk_size,
            self.head_dim,
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen_current_kv_cache_write",
                (
                    aie_utils.get_current_device(),
                    self.packet_seq_len,
                    self.current_slot,
                    self.num_kv_groups,
                    self.q_heads_per_group,
                    self.head_dim,
                    self.chunk_size,
                ),
                {"verbose": mlir_verbose},
            ),
        )

    def get_kernel_artifacts(self):
        return []

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec(
                "in",
                (self.num_kv_groups, self.q_current_elements_per_group),
                dtype=bfloat16,
            ),
            AIERuntimeArgSpec("in", (2,), dtype=bfloat16),
            AIERuntimeArgSpec("out", (self.packet_elements,), dtype=bfloat16),
        ]
