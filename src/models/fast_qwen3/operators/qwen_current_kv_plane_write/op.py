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

from models.fast_qwen3.kv_plane_reference import kv_plane_total_elements


@dataclass
class QwenCurrentKVPlaneWrite(MLIROperator):
    """Write current K/V rows into the FastFlowLM-style four-plane cache.

    The flat physical plane order is ``k03, v03, k47, v47``.  Each token row is
    laid out as ``4 KV heads x head_dim`` inside its plane.
    """

    packet_seq_len: int
    current_slot: int
    q_heads_per_group: int = 2
    head_dim: int = 128
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "packet_seq_len": "psl",
        "current_slot": "slot",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
    }

    def __post_init__(self) -> None:
        if self.packet_seq_len <= 0:
            raise ValueError("packet_seq_len must be positive")
        if self.current_slot < 0 or self.current_slot >= self.packet_seq_len:
            raise ValueError("current_slot must be inside packet_seq_len")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        MLIROperator.__init__(self, context=self.context)

    @property
    def num_kv_groups(self) -> int:
        return 8

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
    def kv_plane_elements(self) -> int:
        return kv_plane_total_elements(self.packet_seq_len, self.head_dim)

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen_current_kv_plane_write",
                (
                    aie_utils.get_current_device(),
                    self.packet_seq_len,
                    self.current_slot,
                    self.q_heads_per_group,
                    self.head_dim,
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
            AIERuntimeArgSpec("inout", (self.kv_plane_elements,), dtype=bfloat16),
        ]
