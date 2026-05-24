# Quantized Qwen3 W4A16

This directory owns the AutoRound W4A16 flow for Qwen3.

## Offline Quantization

Smoke test on CPU:

```bash
uv run python -m models.quantized_qwen3.quantize \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --output-dir ~/models/qwen3-0.6b-w4a16-autogptq \
  --device cpu \
  --iters 0 \
  --nsamples 1 \
  --seqlen 64 \
  --batch-size 1 \
  --disable-opt-rtn
```

Higher-quality calibration should use more samples/iterations and a real
calibration dataset:

```bash
uv run python -m models.quantized_qwen3.quantize \
  Qwen/Qwen3-0.6B \
  --output-dir ~/models/qwen3-0.6b-w4a16-autogptq \
  --device cuda:0 \
  --iters 200 \
  --nsamples 128 \
  --seqlen 512 \
  --dataset ~/datasets/qwen_calibration.json
```

AutoRound first emits an AutoGPTQ-style W4A16 staging checkpoint:

- `*.qweight`: packed int4 weights
- `*.qzeros`: packed zero points
- `*.scales`: per-group scales

`quantize.py` immediately converts that staging checkpoint into the only
supported runtime artifact:

```text
qwen3_w4a16_packed/
  manifest.json
  weights.w4a16.bin
```

This is the runtime format. It is aligned and stores each Linear in two layouts.
The row-major `qparam` layout is used by GEMV decode: each output row stores
biased signed-int4 packed along K followed by that row's bf16 group scales. The
column-stream `gemm_w4_weight` layout is used by fused GEMM:
`(num_aie_columns, n_tile_groups, k_tiles, tile_n // 8, n_block_bytes)`.
Each `8x8` B sub-tile stores packed biased int4 values followed by a 64-lane
bf16 scale vector in the same `s x t` order consumed by `aie::mmul<4,8,8>`.
The AIE kernel loads compressed W4 data, performs tile-local dequantization, and
feeds the result directly into `aie::mmul`; it does not materialize a full bf16
weight tile in DDR.
K/V and gate/up projection pairs are also stored in `paired_linears` as
`(num_aie_columns, n_tile_groups, k_tiles, 2, tile_n // 8, n_block_bytes)`.
This is the format consumed by `W4A16PairedKGroupGEMM`: the pair dimension is
already in the disk artifact, so decode reuses one activation stream without
runtime weight concatenation and accumulates multiple K tiles before storing C.
The `lm_head` GEMM layout pads the vocab dimension to the next 64-wide tile
group across 8 columns and 4 core rows, so vocab projection uses all 32 AIE
cores with the same `tile_n=64` kernel as other Linear layers. The runtime
slices logits back to the real vocabulary size.
Dense non-Linear weights are stored as contiguous bf16 segments in the same
binary. The AutoGPTQ safetensors remain useful only as an offline import/debug
format; inference requires `qwen3_w4a16_packed`.

To pack an existing AutoRound/AutoGPTQ export:

```bash
uv run python -m models.quantized_qwen3.pack \
  ~/models/qwen3-0.6b-w4a16-autogptq
```

## CPU Reference Inference

```bash
uv run python -m models.quantized_qwen3.infer \
  ~/models/qwen3-0.6b-w4a16-autogptq \
  --prompt "The capital of France is" \
  --max-new-tokens 8
```

The current runtime is a PyTorch reference implementation. It reads
`qwen3_w4a16_packed` and runs group-wise fused dequant/GEMM. It is the only
supported inference format in this directory.

## NPU Inference

```bash
uv run python -m models.quantized_qwen3.qwen_npu \
  ~/models/qwen3-0.6b-w4a16-autogptq \
  --prompt-len 64 \
  --num-tokens 8 \
  --batch-size 2
```

The NPU runtime uses one batch transformer decode path for both single-request
and multi-request inference. Transformer decode Linear ops use padded-row
W4A16 fused dequant GEMM over the compressed `gemm_w4_weight` tile stream.
Decode chooses 4/8/16/32 padded rows from the requested batch size, so decode
uses the same batch code path while satisfying the `aie::mmul` tile shape.
Q, attention output, and FFN down projection use `W4A16KGroupGEMM`, which
accumulates multiple K tiles inside one worker call before storing C. Q uses a
larger compile-time K group than output/down because its tile shape fits local
memory and gives better throughput; output/down keep the smaller group that
profiles faster for their shapes. Decode `lm_head` is part of the same fused
ELF and uses `W4A16NShardGEMM` over the compressed W4 tile stream. The vocab/N
dimension is padded for 4-row sharding and spread across AIE columns and core
rows, so single-request decode can use all core rows without padding fake M
rows.
Inside the decode ELF, Q/K RMSNorm and RoPE are fused with `RMSNormRoPE`.
Decode attention keeps each local Q head in AIE vector registers while it
streams packet-cache K/V rows, avoiding repeated Q loads for every KV row.
K/V projection and FFN gate/up projection use `W4A16PairedKGroupGEMM` over the
`paired_linears` W4 tile stream, so each pair shares one activation DMA stream
and reduces C tile load/store traffic across K.
Residual add and the following weighted RMSNorm are fused with
`ResidualAddRMSNorm`, so the updated residual is produced once and streamed
directly into the next norm stage instead of being read back from DDR as the
norm input.
Prefill transformer Linear ops also use `W4A16GEMM`; the final prefill chunk
reloads a final-chunk ELF on the same runtime buffers and produces `logits`
directly from the fused final norm + `lm_head`, so prefill no longer dispatches
a separate vocab projection ELF.
