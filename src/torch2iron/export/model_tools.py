#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import torch


DEFAULT_MODEL_PACKAGE = "models.exported_llama3"


@dataclass(frozen=True)
class ModelExportTools:
    package_name: str
    package_dir: Path
    pytorch_modules: object
    runtime_config: object

    @property
    def config_cls(self):
        config_cls = getattr(self.pytorch_modules, "ExportConfig", None)
        if config_cls is not None:
            return config_cls
        return self.pytorch_modules.LlamaExportConfig

    @property
    def weight_source_aliases(self) -> dict[str, str]:
        return dict(getattr(self.pytorch_modules, "WEIGHT_SOURCE_ALIASES", {}))


def load_model_export_tools(model_package: str = DEFAULT_MODEL_PACKAGE) -> ModelExportTools:
    package = importlib.import_module(model_package)
    package_file = getattr(package, "__file__", None)
    if package_file is None:
        raise RuntimeError(f"{model_package} is not a filesystem package")

    return ModelExportTools(
        package_name=model_package,
        package_dir=Path(package_file).resolve().parent,
        pytorch_modules=importlib.import_module(f"{model_package}.pytorch_modules"),
        runtime_config=importlib.import_module(f"{model_package}.runtime_config"),
    )


def make_llama_export_config(
    tools: ModelExportTools,
    *,
    layers: int,
    heads: int,
    kv_groups: int,
    head_dim: int,
    hidden_dim: int,
    vocab_size: int,
    max_seq_len: int,
    chunk_size: int,
):
    return tools.config_cls(
        vocab_size=vocab_size,
        emb_dim=heads * head_dim,
        n_layers=layers,
        n_heads=heads,
        n_kv_groups=kv_groups,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
    )


def export_program(
    tools: ModelExportTools,
    config,
    mode: str,
) -> torch.export.ExportedProgram:
    if mode == "prefill":
        model_cls = getattr(tools.pytorch_modules, "ExportPrefillModel", None)
        if model_cls is None:
            model_cls = tools.pytorch_modules.ExportLlamaPrefillModel
        model = model_cls(config).eval()
        args = tools.pytorch_modules.example_prefill_args(config)
    elif mode == "decode":
        model_cls = getattr(tools.pytorch_modules, "ExportDecodeModel", None)
        if model_cls is None:
            model_cls = tools.pytorch_modules.ExportLlamaDecodeModel
        model = model_cls(config).eval()
        args = tools.pytorch_modules.example_decode_args(config)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    exported_program: torch.export.ExportedProgram = torch.export.export(model, args)
    return exported_program
