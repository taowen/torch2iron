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
column-stream `gemm_weight` layout is used by fused GEMM:
`(num_aie_columns, n_tile_groups, k_tiles, tile_k // 8, tile_n // 8, 8, 8)`.
It stores bf16 weights that were dequantized once during packing and arranged
in the `s x t` subtile order consumed by `aie::mmul<4,8,8>`, so the AIE kernel
does not spend its inner loop on qparam unpack, bias subtraction, scaling, or
weight transposition.
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
`W4A16GEMM` over the pre-dequantized `gemm_weight` tile stream. Decode chooses
4/8/16/32 padded rows from the requested batch size, so decode uses the same
batch code path while satisfying the `aie::mmul` tile shape. The `lm_head` uses
a batch `W4A16GEMV` that shares one `uint8` qparam buffer across all active
batch lanes, avoiding qparam replication while keeping the generated IR small.
Prefill transformer Linear ops also use `W4A16GEMM`, so the hot paths do not
build temporary dense bf16 Linear weights on the host.
