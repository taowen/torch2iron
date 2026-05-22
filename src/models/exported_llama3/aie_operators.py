#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedFullELFCallable

from models.exported_llama3.generated.decode_fused import build_decode_fused_op
from models.exported_llama3.generated.prefill_operators import (
    build_prefill_operations,
)
from models.exported_llama3.llama_packed_weights import (
    load_llama_packed_segment,
    validate_llama_packed_weight_artifact,
)
from models.exported_llama3.llama_weight_layout import iter_llama_decode_weight_specs


class AIEPrefillOperations:
    pass


class AIEDecodeOperations:
    pass


class AIELlamaOperators:
    def __init__(self, config, prompt_len):
        build_suffix = f"seq{prompt_len}"
        self.context = AIEContext(build_dir=Path("build") / build_suffix)
        self.context.build_dir.mkdir(parents=True, exist_ok=True)

        self.prefill = AIEPrefillOperations()
        self.decode = AIEDecodeOperations()

        self._build_prefill_ops(config, prompt_len)
        self._build_decode_ops(config, prompt_len, build_suffix)

    def _build_prefill_ops(self, config, prompt_len):
        build_prefill_operations(self.prefill, config, prompt_len, self.context)

    def _build_decode_ops(self, config, prompt_len, build_suffix):
        self.decode.fused_op, self.decode.current_cache_slot = build_decode_fused_op(
            config, prompt_len, build_suffix
        )
        self.decode.fused = FusedFullELFCallable(self.decode.fused_op)

        load_decode_weight_buffers(config, self.decode.fused)
        self.decode.fused.input_buffer.to("npu")
        self.decode.fused.weight_buffer.to("npu")
        self.decode.fused.lm_head_buffer.to("npu")
        self.decode.fused.scratch_buffer.to("npu")
        self.decode.fused.output_buffer.to("npu")


def load_decode_weight_buffers(config, fused):
    packed_dir = getattr(config, "packed_weights_dir", None)
    require_packed = getattr(config, "require_packed_weights", False)
    manifest = None
    if packed_dir is not None and Path(packed_dir).exists():
        manifest = validate_llama_packed_weight_artifact(config, packed_dir)
        logging.info("Loading decode weights from packed artifact: %s", packed_dir)
    elif require_packed:
        raise FileNotFoundError(f"packed weights required but not found at {packed_dir}")
    else:
        logging.info("Loading decode weights from safetensors tensors")

    for spec in iter_llama_decode_weight_specs(config):
        view = fused.get_buffer(spec["name"]).torch_view()
        if manifest is None:
            view[:] = config.weights[spec["source"]].flatten()
        else:
            view[:] = load_llama_packed_segment(
                packed_dir, manifest, spec["name"]
            ).flatten()
