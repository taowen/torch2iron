#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedFullELFCallable, FusedMLIROperator
from torch2iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMM,
    GEMV,
    LlamaChunkedAttention,
    RMSNorm,
    RoPE,
    SiLU,
    StridedCopy,
)

from models.exported_llama3.llama_weight_layout import (
    iter_llama_decode_weight_specs,
    llama_decode_lm_head_weight_names,
    llama_decode_transformer_weight_names,
    load_llama_packed_segment,
    validate_llama_packed_weight_artifact,
)
from models.exported_llama3.runtime_config import DECODE_ATTN_CHUNK_SIZE


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
        self.prefill.rms_norm = (
            RMSNorm(
                size=prompt_len * config.emb_dim,
                num_aie_columns=8,
                num_channels=1,
                tile_size=config.emb_dim,
                weighted=True,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.residual_add = (
            ElementwiseAdd(
                size=prompt_len * config.emb_dim,
                tile_size=config.emb_dim,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        min_N = 64 * 8 * 4  # tile_n * num_aie_columns * partition_N
        config.padded_vocab_size = (config.vocab_size + min_N - 1) // min_N * min_N
        config.vocab_partitions = 4
        self.prefill.gemv_out_head_compilable = GEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.padded_vocab_size // config.vocab_partitions,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=True,
            separate_c_tiles=True,
            context=self.context,
        ).compile()
        self.prefill.out_head = self.prefill.gemv_out_head_compilable.get_callable()

        self.prefill.ffn_up_gate = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.hidden_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.ffn_down = (
            GEMM(
                M=prompt_len,
                K=config.hidden_dim,
                N=config.emb_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.ffn_silu = (
            SiLU(
                size=prompt_len * config.hidden_dim,
                tile_size=config.hidden_dim,
                num_aie_columns=8,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.eltwise_mul_ffn = (
            ElementwiseMul(
                size=prompt_len * config.hidden_dim,
                tile_size=config.hidden_dim,
                num_aie_columns=8,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_scale = (
            ElementwiseMul(
                size=config.n_heads * prompt_len * prompt_len,
                tile_size=prompt_len,
                num_aie_columns=8,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.rope_queries = (
            RoPE(
                rows=prompt_len * config.n_heads,
                cols=config.head_dim,
                angle_rows=prompt_len,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.rope_keys = (
            RoPE(
                rows=prompt_len * config.n_kv_groups,
                cols=config.head_dim,
                angle_rows=prompt_len,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_query = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.n_heads * config.head_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_key = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.n_kv_groups * config.head_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_value = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.n_kv_groups * config.head_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_scores = (
            GEMM(
                M=prompt_len,
                K=config.head_dim,
                N=prompt_len,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

    def _build_decode_ops(self, config, prompt_len, build_suffix):
        elf_ctx = AIEContext(build_dir=Path("build_elf") / build_suffix)

        gemv_attn_query_op = GEMV(
            M=config.n_heads * config.head_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.head_dim // 2,
            context=elf_ctx,
        )

        gemv_attn_key_value_op = GEMV(
            M=config.n_kv_groups * config.head_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.head_dim // 2,
            context=elf_ctx,
        )

        rope_queries_op = RoPE(
            rows=config.n_heads, cols=config.head_dim, angle_rows=1, context=elf_ctx
        )
        rope_keys_op = RoPE(
            rows=config.n_kv_groups,
            cols=config.head_dim,
            angle_rows=1,
            context=elf_ctx,
        )

        self.decode.current_cache_slot = prompt_len - 1
        packet_chunk_elements = (
            2 * DECODE_ATTN_CHUNK_SIZE * config.head_dim + DECODE_ATTN_CHUNK_SIZE
        )
        packet_elements_per_group = (
            prompt_len // DECODE_ATTN_CHUNK_SIZE * packet_chunk_elements
        )
        packet_elements = config.n_kv_groups * packet_elements_per_group
        current_chunk = self.decode.current_cache_slot // DECODE_ATTN_CHUNK_SIZE
        current_row = self.decode.current_cache_slot % DECODE_ATTN_CHUNK_SIZE
        current_k_packet_offset = (
            current_chunk * packet_chunk_elements + current_row * config.head_dim
        )
        current_v_packet_offset = (
            current_chunk * packet_chunk_elements
            + DECODE_ATTN_CHUNK_SIZE * config.head_dim
            + current_row * config.head_dim
        )

        strided_copy_packet_key_op = StridedCopy(
            input_sizes=(config.n_kv_groups, config.head_dim),
            input_strides=(config.head_dim, 1),
            input_offset=0,
            output_sizes=(config.n_kv_groups, config.head_dim),
            output_strides=(packet_elements_per_group, 1),
            output_offset=current_k_packet_offset,
            input_buffer_size=config.n_kv_groups * config.head_dim,
            output_buffer_size=packet_elements,
            num_aie_channels=1,
            context=elf_ctx,
        )

        strided_copy_packet_value_op = StridedCopy(
            input_sizes=(config.n_kv_groups, config.head_dim),
            input_strides=(config.head_dim, 1),
            input_offset=0,
            output_sizes=(config.n_kv_groups, config.head_dim),
            output_strides=(packet_elements_per_group, 1),
            output_offset=current_v_packet_offset,
            input_buffer_size=config.n_kv_groups * config.head_dim,
            output_buffer_size=packet_elements,
            num_aie_channels=1,
            context=elf_ctx,
        )

        copy_present_kv_op = StridedCopy(
            input_sizes=(config.n_kv_groups, config.head_dim),
            input_strides=(config.head_dim, 1),
            input_offset=0,
            output_sizes=(config.n_kv_groups, config.head_dim),
            output_strides=(config.head_dim, 1),
            output_offset=0,
            input_buffer_size=config.n_kv_groups * config.head_dim,
            output_buffer_size=config.n_kv_groups * config.head_dim,
            num_aie_channels=1,
            context=elf_ctx,
        )

        llama_chunked_attention_op = LlamaChunkedAttention(
            max_seq_len=prompt_len,
            num_kv_groups=config.n_kv_groups,
            q_heads_per_group=config.n_heads // config.n_kv_groups,
            head_dim=config.head_dim,
            chunk_size=DECODE_ATTN_CHUNK_SIZE,
            context=elf_ctx,
        )

        gemv_attn_output_op = GEMV(
            M=config.emb_dim,
            K=config.n_heads * config.head_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.emb_dim // 8,
            context=elf_ctx,
        )

        rms_norm_op = RMSNorm(
            size=config.emb_dim,
            num_aie_columns=1,
            num_channels=1,
            tile_size=config.emb_dim,
            weighted=True,
            context=elf_ctx,
        )

        gemv_ffn_up_gate_op = GEMV(
            M=config.hidden_dim,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=config.hidden_dim // 8,
            context=elf_ctx,
        )

        gemv_ffn_down_op = GEMV(
            M=config.emb_dim,
            K=config.hidden_dim,
            num_aie_columns=8,
            tile_size_input=1,
            tile_size_output=config.emb_dim // 8,
            context=elf_ctx,
        )

        silu_ffn_op = SiLU(
            size=config.hidden_dim,
            tile_size=config.hidden_dim // 8,
            num_aie_columns=8,
            context=elf_ctx,
        )

        eltwise_mul_ffn_op = ElementwiseMul(
            size=config.hidden_dim,
            tile_size=config.hidden_dim // 8,
            num_aie_columns=8,
            context=elf_ctx,
        )

        residual_add_op = ElementwiseAdd(
            size=config.emb_dim, tile_size=config.emb_dim // 8, context=elf_ctx
        )

        gemv_out_head_op = GEMV(
            M=config.vocab_size,
            K=config.emb_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=32,
            context=elf_ctx,
        )

        packet_cache_buffer_size = packet_elements * 2  # bfloat16 byte size
        runlist = []
        for layer_idx in range(config.n_layers):
            runlist.extend(
                [
                    (rms_norm_op, "x", f"W_norm1_{layer_idx}", "x_norm"),
                    (
                        gemv_attn_query_op,
                        f"W_attn_query_{layer_idx}",
                        "x_norm",
                        "queries",
                    ),
                    (
                        gemv_attn_key_value_op,
                        f"W_attn_key_{layer_idx}",
                        "x_norm",
                        "keys",
                    ),
                    (
                        gemv_attn_key_value_op,
                        f"W_attn_value_{layer_idx}",
                        "x_norm",
                        "values",
                    ),
                    (rope_queries_op, "queries", "rope_angles", "queries"),
                    (rope_keys_op, "keys", "rope_angles", "keys"),
                    (copy_present_kv_op, "keys", f"present_keys_{layer_idx}"),
                    (copy_present_kv_op, "values", f"present_values_{layer_idx}"),
                    (
                        strided_copy_packet_key_op,
                        "keys",
                        f"packet_cache_{layer_idx}",
                    ),
                    (
                        strided_copy_packet_value_op,
                        "values",
                        f"packet_cache_{layer_idx}",
                    ),
                    (
                        llama_chunked_attention_op,
                        "queries",
                        f"packet_cache_{layer_idx}",
                        "attn_context",
                    ),
                    (
                        gemv_attn_output_op,
                        f"W_attn_output_decode_{layer_idx}",
                        "attn_context",
                        "attn_output",
                    ),
                    (residual_add_op, "x", "attn_output", "x"),
                    (rms_norm_op, "x", f"W_norm2_{layer_idx}", "x_norm"),
                    (
                        gemv_ffn_up_gate_op,
                        f"W_ffn_gate_{layer_idx}",
                        "x_norm",
                        "ffn_gate",
                    ),
                    (gemv_ffn_up_gate_op, f"W_ffn_up_{layer_idx}", "x_norm", "ffn_up"),
                    (silu_ffn_op, "ffn_gate", "ffn_gate"),
                    (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
                    (
                        gemv_ffn_down_op,
                        f"W_ffn_down_{layer_idx}",
                        "ffn_hidden",
                        "ffn_output",
                    ),
                    (residual_add_op, "x", "ffn_output", "x"),
                ]
            )
        runlist += [
            (rms_norm_op, "x", "W_final_norm", "hidden_out"),
            (gemv_out_head_op, "W_out_head", "hidden_out", "logits"),
        ]

        self.decode.fused_op = FusedMLIROperator(
            "fused_op",
            runlist,
            input_args=["x", "rope_angles"],
            output_args=[
                "logits",
                *[f"present_keys_{layer_idx}" for layer_idx in range(config.n_layers)],
                *[
                    f"present_values_{layer_idx}"
                    for layer_idx in range(config.n_layers)
                ],
            ],
            buffer_sizes={
                f"packet_cache_{layer_idx}": packet_cache_buffer_size
                for layer_idx in range(config.n_layers)
            },
            external_args={
                "weight": llama_decode_transformer_weight_names(config),
                "lm_head": llama_decode_lm_head_weight_names(config),
            },
            context=elf_ctx,
        ).compile()

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
