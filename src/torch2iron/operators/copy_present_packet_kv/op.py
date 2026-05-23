# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import aie.utils as aie_utils
from ml_dtypes import bfloat16

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
)


@dataclass
class CopyPresentPacketKV(MLIROperator):
    """Copy decode K/V into present outputs and the current packet-cache slot."""

    kv_dim: int
    num_kv_groups: int
    head_dim: int
    packet_elements: int
    packet_elements_per_group: int
    key_packet_offset: int
    value_packet_offset: int
    dtype: object = field(default=bfloat16, repr=False)
    transfer_size: int | None = None
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "kv_dim": "kv",
        "num_kv_groups": "g",
        "head_dim": "h",
        "packet_elements": "pe",
        "packet_elements_per_group": "peg",
        "key_packet_offset": "ko",
        "value_packet_offset": "vo",
        "transfer_size": "tr",
    }

    def __post_init__(self):
        if self.kv_dim != self.num_kv_groups * self.head_dim:
            raise ValueError("kv_dim must equal num_kv_groups * head_dim")
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "copy_present_packet_kv",
                (
                    aie_utils.get_current_device(),
                    self.dtype,
                    self.kv_dim,
                    self.num_kv_groups,
                    self.head_dim,
                    self.packet_elements,
                    self.packet_elements_per_group,
                    self.key_packet_offset,
                    self.value_packet_offset,
                    self.transfer_size,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return []

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", self.kv_dim),
            AIERuntimeArgSpec("in", self.kv_dim),
            AIERuntimeArgSpec("out", self.kv_dim),
            AIERuntimeArgSpec("out", self.kv_dim),
            AIERuntimeArgSpec("out", self.packet_elements),
        ]
