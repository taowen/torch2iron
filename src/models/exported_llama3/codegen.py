#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate exported_llama3 runtime code directly from torch.export graphs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from models.exported_llama3.dump_exported_program import export_program
from models.exported_llama3.pytorch_modules import LlamaExportConfig
from models.exported_llama3.runtime_config import (
    DECODE_ATTN_CHUNK_SIZE,
    MIN_COMPILED_SEQ_LEN,
)


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def call_function_nodes(exported_program):
    return [
        node
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def module_paths(node):
    stack = node.meta.get("nn_module_stack") or {}
    return [
        entry[0]
        for entry in stack.values()
        if isinstance(entry, tuple) and entry
    ]


def path_endswith(node, suffix: str) -> bool:
    return any(path.endswith(suffix) for path in module_paths(node))


def layer_idx(node) -> int | None:
    for path in module_paths(node):
        match = _LAYER_RE.search(path)
        if match:
            return int(match.group(1))
    return None


def layer_indices(exported_program) -> list[int]:
    return sorted(
        {
            layer_idx(node)
            for node in call_function_nodes(exported_program)
            if layer_idx(node) is not None
        }
    )


def representative_layer_nodes(exported_program) -> list[object]:
    layers = layer_indices(exported_program)
    if not layers:
        raise RuntimeError("exported graph has no transformer layer nodes")
    representative_layer = layers[0]
    return [
        node
        for node in call_function_nodes(exported_program)
        if layer_idx(node) == representative_layer
    ]


def target_is(node, target: str) -> bool:
    return str(node.target) == target


def tensor_rank(node) -> int:
    tensor_meta = node.meta.get("tensor_meta")
    return len(getattr(tensor_meta, "shape", ()))


def arg_path_endswith(node, arg_idx: int, suffix: str) -> bool:
    if len(node.args) <= arg_idx:
        return False
    arg = node.args[arg_idx]
    return hasattr(arg, "meta") and path_endswith(arg, suffix)


def _decode_chunked_attention_nodes(exported_program):
    return [
        node
        for node in call_function_nodes(exported_program)
        if target_is(node, "torch2iron.llama_chunked_attention.default")
    ]


def decode_max_seq_len(exported_program) -> int:
    nodes = _decode_chunked_attention_nodes(exported_program)
    if not nodes:
        raise RuntimeError("decode graph has no llama_chunked_attention node")
    return int(nodes[0].args[4])


def decode_chunk_size(exported_program) -> int:
    nodes = _decode_chunked_attention_nodes(exported_program)
    if not nodes:
        raise RuntimeError("decode graph has no llama_chunked_attention node")
    return int(nodes[0].args[5])


def decode_buffer_name(node) -> str | None:
    layer = layer_idx(node)
    if target_is(node, "aten.rms_norm.default") and path_endswith(
        node, "input_layernorm"
    ):
        return f"W_norm1_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(
        node, "self_attn.q_proj"
    ):
        return f"W_attn_query_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(
        node, "self_attn.k_proj"
    ):
        return f"W_attn_key_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(
        node, "self_attn.v_proj"
    ):
        return f"W_attn_value_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(
        node, "self_attn.o_proj"
    ):
        return f"W_attn_output_decode_{layer}"
    if target_is(node, "aten.rms_norm.default") and path_endswith(
        node, "post_attention_layernorm"
    ):
        return f"W_norm2_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(
        node, "mlp.gate_proj"
    ):
        return f"W_ffn_gate_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(node, "mlp.up_proj"):
        return f"W_ffn_up_{layer}"
    if target_is(node, "aten.linear.default") and path_endswith(
        node, "mlp.down_proj"
    ):
        return f"W_ffn_down_{layer}"
    if (
        target_is(node, "aten.rms_norm.default")
        and path_endswith(node, "norm")
        and layer is None
    ):
        return "W_final_norm"
    if target_is(node, "aten.linear.default") and path_endswith(node, "lm_head"):
        return "W_out_head"
    return None


def decode_transformer_weight_names(exported_program) -> list[str]:
    names = []
    for node in call_function_nodes(exported_program):
        name = decode_buffer_name(node)
        if name is not None and name != "W_out_head":
            names.append(name)
    return names


def decode_lm_head_weight_names(exported_program) -> list[str]:
    names = []
    for node in call_function_nodes(exported_program):
        name = decode_buffer_name(node)
        if name == "W_out_head":
            names.append(name)
    return names


def _parameter_targets(exported_program) -> dict[str, str]:
    return {
        spec.arg.name: spec.target
        for spec in exported_program.graph_signature.input_specs
        if spec.target is not None
    }


def decode_weight_source(exported_program, node) -> str | None:
    if target_is(node, "aten.linear.default"):
        parameter_node = node.args[1]
    elif target_is(node, "aten.rms_norm.default"):
        parameter_node = node.args[2]
    else:
        return None

    return _parameter_targets(exported_program)[parameter_node.name]


def decode_weight_specs(exported_program) -> list[dict[str, object]]:
    specs = []
    for node in call_function_nodes(exported_program):
        name = decode_buffer_name(node)
        if name is None:
            continue
        group = "lm_head" if name == "W_out_head" else "weight"
        specs.append(
            {
                "layer": layer_idx(node),
                "group": group,
                "name": name,
                "source": decode_weight_source(exported_program, node),
            }
        )
    return specs


def export_decode_program_for_codegen() -> object:
    config = LlamaExportConfig(
        vocab_size=128,
        emb_dim=32,
        n_layers=16,
        n_heads=4,
        n_kv_groups=2,
        head_dim=8,
        hidden_dim=64,
        max_seq_len=MIN_COMPILED_SEQ_LEN,
        chunk_size=DECODE_ATTN_CHUNK_SIZE,
    )
    return export_program(config, "decode")


def export_prefill_program_for_codegen() -> object:
    config = LlamaExportConfig(
        vocab_size=128,
        emb_dim=32,
        n_layers=16,
        n_heads=4,
        n_kv_groups=2,
        head_dim=8,
        hidden_dim=64,
        max_seq_len=8,
        chunk_size=DECODE_ATTN_CHUNK_SIZE,
    )
    return export_program(config, "prefill")


def _jinja_env() -> Environment:
    template_dir = Path(__file__).with_name("templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(
        call_function_nodes=call_function_nodes,
        path_endswith=path_endswith,
        layer_idx=layer_idx,
        layer_indices=layer_indices,
        representative_layer_nodes=representative_layer_nodes,
        target_is=target_is,
        tensor_rank=tensor_rank,
        arg_path_endswith=arg_path_endswith,
        decode_max_seq_len=decode_max_seq_len,
        decode_chunk_size=decode_chunk_size,
        decode_transformer_weight_names=decode_transformer_weight_names,
        decode_lm_head_weight_names=decode_lm_head_weight_names,
        decode_weight_specs=decode_weight_specs,
    )
    return env


def render_decode_layout(exported_program) -> str:
    return _jinja_env().get_template("decode_layout.py.j2").render(
        exported_program=exported_program
    )


def render_decode_fused(exported_program) -> str:
    env = _jinja_env()
    return env.get_template("decode_fused.py.j2").render(
        exported_program=exported_program
    )


def render_prefill_runtime(exported_program) -> str:
    return _jinja_env().get_template("prefill_runtime.py.j2").render(
        exported_program=exported_program
    )


def render_generated_files() -> dict[str, str]:
    decode_program = export_decode_program_for_codegen()
    prefill_program = export_prefill_program_for_codegen()
    return {
        "decode_layout.py": render_decode_layout(decode_program),
        "decode_fused.py": render_decode_fused(decode_program),
        "prefill_runtime.py": render_prefill_runtime(prefill_program),
    }


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate exported_llama3 runtime code from torch.export graphs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("generated"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the generated file is current without rewriting it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered_files = render_generated_files()
    if args.check:
        for name, rendered in rendered_files.items():
            output = args.output_dir / name
            current = output.read_text() if output.exists() else None
            if current != rendered:
                raise SystemExit(f"{output} is not up to date")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rendered in rendered_files.items():
        (args.output_dir / name).write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
