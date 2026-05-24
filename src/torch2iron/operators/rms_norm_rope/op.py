# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
class RMSNormRoPE(MLIROperator):
    """Weighted RMSNorm followed by two-halves RoPE for decode Q/K heads."""

    rows: int
    cols: int
    angle_rows: int
    num_aie_columns: int = 8
    epsilon: float = 1e-5
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "angle_rows": "arows",
    }

    def __post_init__(self):
        if self.cols % 32 != 0:
            raise ValueError("cols must be divisible by 32")
        if self.rows % self.num_aie_columns != 0:
            raise ValueError("rows must be divisible by num_aie_columns")
        if self.angle_rows > self.rows or self.rows % self.angle_rows != 0:
            raise ValueError("angle_rows must divide rows")
        if self.angle_rows < self.num_aie_columns or self.angle_rows % self.num_aie_columns != 0:
            raise ValueError("angle_rows must be divisible by num_aie_columns")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _kernel_object(self):
        return f"weighted_rms_norm_rope_{self._epsilon_tag}.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "weighted_rms_norm_rope",
                (
                    aie_utils.get_current_device(),
                    self.rows,
                    self.cols,
                    self.angle_rows,
                    self.num_aie_columns,
                    0,
                ),
                {"kernel_object": self._kernel_object},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_object,
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.rows, self.cols)),
            AIERuntimeArgSpec("in", (self.cols,)),
            AIERuntimeArgSpec("in", (self.angle_rows, self.cols)),
            AIERuntimeArgSpec("out", (self.rows, self.cols)),
        ]
