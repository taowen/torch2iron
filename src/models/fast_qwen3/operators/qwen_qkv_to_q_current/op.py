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

from models.fast_qwen3.q4nx_layout import Q4NX_PATCH_OUT_ROWS


@dataclass
class QwenQKVToQCurrent(MLIROperator):
    """Assemble Q/K/V projection patches into grouped ``q_current``.

    Input layout is the QKV projection patch output:
    ``[qkv_output_patches, 3, 64]`` flattened with projection order Q, K, V.
    Output layout is per KV group:
    ``[q_heads_per_group * head_dim, current_key, current_value]``.
    """

    qkv_output_patches: int = 8
    num_kv_groups: int = 2
    q_heads_per_group: int = 2
    head_dim: int = 128
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "qkv_output_patches": "opatches",
        "num_kv_groups": "kvg",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
    }

    def __post_init__(self) -> None:
        if self.qkv_output_patches <= 0:
            raise ValueError("qkv_output_patches must be positive")
        if self.num_kv_groups <= 0:
            raise ValueError("num_kv_groups must be positive")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % Q4NX_PATCH_OUT_ROWS != 0:
            raise ValueError(f"head_dim must be divisible by {Q4NX_PATCH_OUT_ROWS}")
        required_q_patches = (
            self.num_kv_groups
            * self.q_heads_per_group
            * self.head_chunks
        )
        required_kv_patches = self.num_kv_groups * self.head_chunks
        if self.qkv_output_patches < required_q_patches:
            raise ValueError(
                f"qkv_output_patches={self.qkv_output_patches} cannot cover "
                f"{required_q_patches} Q patches"
            )
        if self.qkv_output_patches < required_kv_patches:
            raise ValueError(
                f"qkv_output_patches={self.qkv_output_patches} cannot cover "
                f"{required_kv_patches} K/V patches"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def head_chunks(self) -> int:
        return self.head_dim // Q4NX_PATCH_OUT_ROWS

    @property
    def qkv_elements(self) -> int:
        return self.qkv_output_patches * 3 * Q4NX_PATCH_OUT_ROWS

    @property
    def q_elements_per_group(self) -> int:
        return self.q_heads_per_group * self.head_dim

    @property
    def q_current_elements_per_group(self) -> int:
        return self.q_elements_per_group + 2 * self.head_dim

    @property
    def output_elements(self) -> int:
        return self.num_kv_groups * self.q_current_elements_per_group

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen_qkv_to_q_current",
                (
                    aie_utils.get_current_device(),
                    self.qkv_output_patches,
                    self.num_kv_groups,
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
            AIERuntimeArgSpec("in", (self.qkv_elements,), dtype=bfloat16),
            AIERuntimeArgSpec("out", (self.output_elements,), dtype=bfloat16),
        ]
