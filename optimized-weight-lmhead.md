# Optimized Weight and LM Head Plan

## Background

The decode path now uses a fused Llama body with chunked KV attention. The next
large bottleneck is no longer the attention scratch traffic alone. It is the way
weights are loaded and placed into the fused runtime.

The current `FusedMLIROperator` exposes three runtime buffers:

```text
input_buffer
output_buffer
scratch_buffer
```

Any buffer that is not declared as an input or output is placed in
`scratch_buffer`. In the fused Llama decode body, that means transformer weights
are part of the scratch BO.

Before moving LM head to CPU, the single fused scratch BO was about `2.49GB`:

```text
16 transformer layers bf16 weights   ~1.95GB
LM head bf16 weight                  ~0.53GB
packet KV cache                      ~0.02GB
intermediate buffers/alignment       small
```

After moving LM head out of the fused scratch, decode can run because the
single scratch BO drops below the current XRT host BO / memlock limit. The final
target is stronger: each decode token should still use one fused ELF dispatch,
and LM head should run inside that same ELF, but transformer weights and LM head
weights must live in external BOs rather than the scratch BO.

## What IRON Does

The Qwen3 persistent path in `~/projects/IRON` does not treat safetensors as the
runtime layout. Safetensors remain the source of truth, but the fast path can
prepare a separate packed artifact:

```text
<model_dir>/qwen3_iron_packed/
  weights.bf16.bin
  manifest.json
```

The artifact has two important properties:

- `weights.bf16.bin` is a contiguous raw bf16 file ordered for the runtime graph.
- `manifest.json` records per-layer and per-segment offsets, sizes, shapes, and
  model config metadata.

At runtime, IRON loads the packed file once, creates one parent `XRTTensor`, and
uses `XRTSubBuffer` views for per-layer or per-chunk weight arguments. This
avoids per-token/per-layer packing and avoids many independent weight BO
allocations.

The key lesson is:

```text
safetensors -> packed runtime artifact -> parent XRT weight BO -> sub-buffer views
```

Mmap is useful as a loader implementation detail, but it is not the core
optimization. The core optimization is making the disk format match the runtime
access pattern and avoiding a giant mixed scratch BO.

## Current Llama Problem

The current Llama path does this instead:

```text
safetensors
  -> PyTorch tensors in config.weights
  -> per-buffer copy into fused scratch sub-buffers
  -> one large scratch BO containing weights + cache + intermediates
```

This has three issues:

1. Weight layout is not persisted in the form the NPU runtime consumes.
2. Startup repeats many large copies into scratch sub-buffers.
3. Scratch BO size is coupled to model weight size.

Converting safetensors to a more mmap-friendly file only solves part of issue 1.
It does not solve issue 3 unless the fused runtime can consume weights from a
separate weight BO.

## Target Weight Artifact

Add a Llama-specific packed artifact:

```text
<model_dir>/llama_iron_packed/
  weights.bf16.bin
  manifest.json
```

For the current decode body, each layer should be packed in the order the fused
runlist consumes weights:

```text
input_layernorm.weight
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
self_attn.o_proj.weight
post_attention_layernorm.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

The manifest should include:

```text
format
dtype
element_size_bytes
weight_order
model_config
num_layers
per_layer_numel
total_numel
total_bytes
data_file
layers[id, element_offset, numel, byte_offset, byte_length, segments]
```

Use bf16 raw storage first. Do not introduce block-f16 or quantized weights
until the bf16 path is proven correct, because the current GEMV kernels already
consume ordinary contiguous bf16 layouts.

## Runtime Design

There are three possible implementation levels.

### Level 1: Packed Artifact Only

Load `weights.bf16.bin` and copy slices into the existing fused scratch
sub-buffers.

This improves startup determinism and removes repeated layout packing, but it
does not reduce scratch BO size. It is useful as an intermediate correctness
step, not the final optimization.

### Level 2: External Weight BOs

Extend the fused runtime to support external buffers:

```text
input_buffer
output_buffer
scratch_buffer
weight_buffer
lm_head_buffer
```

Then place transformer weights in `weight_buffer` and LM head weights in
`lm_head_buffer` instead of `scratch_buffer`. The NPU sequence receives five BO
arguments, and fused sub-buffer layout has explicit external buffer classes:

```text
input
output
scratch
weight
lm_head
```

The Llama decode setup would:

1. Load or create `llama_iron_packed/weights.bf16.bin`.
2. Create one parent `XRTTensor` for transformer weights.
3. Create one parent `XRTTensor` for LM head weights.
4. Bind every `W_*_{layer}` fused buffer name to a view into `weight_buffer`.
5. Bind `W_out_head` to a view into `lm_head_buffer`.
6. Keep packet KV cache and intermediates in `scratch_buffer`.

This is the preferred target because it keeps the current single-dispatch fused
decode structure while decoupling scratch size from model size.

Expected result:

```text
weight BO  ~1.95GB for transformer weights
lm_head BO ~0.53GB
scratch BO ~17MB for seq512, dominated by packet cache and intermediates
one fused ELF dispatch per decoded token
```

This may not reduce total resident memory, but it avoids one oversized mixed BO,
keeps each BO below the current single-allocation limit, and makes memory
ownership explicit.

### Level 3: Layer-Chunk Runtime

Follow the Qwen3 persistent model more closely: compile an n-layer or layer-chunk
operator whose weight argument is a packed layer/chunk tensor, then pass
`XRTSubBuffer` slices from the parent weight BO.

This is more flexible and naturally supports layer chunking, but it is a larger
rewrite. It is useful if the four-BO fused runtime hits compiler or XRT limits.

## LM Head Policy

Keep LM head out of fused scratch, but run it inside the same decode ELF.

For Llama 3.2 1B, tied embedding / LM head is:

```text
vocab_size x hidden_size x bf16
= 128256 x 2048 x 2 bytes
~= 525MB
```

Putting this inside fused scratch was enough to push the single BO to about
`2.49GB`. Placing it in a separate `lm_head_buffer` keeps decode as a single ELF
dispatch without recreating the oversized scratch BO.

The current decode path should remain:

```text
NPU fused transformer body -> hidden_out
NPU LM head in the same fused ELF -> logits
CPU sampling
```

Future LM head optimizations should be separate from transformer weight BO work:

- prepack LM head as its own artifact segment;
- keep LM head in its own runtime BO even when it is called by the fused ELF;
- add CPU-side quantized/int8 LM head;
- compute only candidate vocab partitions if a sampling strategy allows it.

Do not reinsert LM head into transformer fused scratch.

## Mmap Decision

Mmap is worth using only after the artifact format is defined.

Recommended first implementation:

```text
np.fromfile(..., dtype=np.uint16).copy() -> torch.bfloat16
```

This matches the proven IRON Qwen3 path and avoids lifetime surprises. After the
packed artifact and weight BO path are correct, switch the loader to
`np.memmap` or `torch.from_file` if startup memory pressure is measurable.

Important distinction:

```text
mmap safetensors                 not enough
mmap packed runtime artifact     useful
external weight BO               required to shrink fused scratch
```

## Implementation Plan

1. Add Llama packed weight layout helpers.

   Create helpers near the Llama model path or a small shared layout module:

   ```text
   default_llama_packed_weights_dir(model_path)
   pack_llama_layer_weights(config, layer_idx)
   write_llama_packed_weight_artifact(config, output_dir)
   load_llama_packed_weight_tensor(output_dir, manifest)
   validate_llama_packed_weight_artifact(config, output_dir)
   llama_packed_weight_layer_slice(...)
   ```

2. Add CLI flags.

   ```text
   --prepare-weights
   --packed-weights-dir
   --require-packed-weights
   ```

   `--prepare-weights` should write the artifact and exit. Normal decode can
   still fall back to packing from safetensors-derived tensors.

3. Prove Level 1.

   Load the packed artifact and copy its segments into the existing fused scratch
   sub-buffers. Verify that short prompts still match the CPU reference.

4. Add external weight BO support.

   Extend `FusedMLIROperator` and `FusedFullELFCallable` so selected buffer names
   are assigned to external buffers. Update fused MLIR generation to pass runtime
   sequence arguments for `weight_buffer` and `lm_head_buffer`.

5. Bind Llama weights to external BOs.

   Replace per-buffer scratch writes for `W_*` with one packed parent weight BO
   and manifest-driven sub-buffer bindings. Bind `W_out_head` to `lm_head_buffer`.

6. Keep sampling on CPU and measure separately.

   Track:

   ```text
   decode_npu_time
   fused_input_sync_s
   fused_output_sync_s
   cpu_lm_head_s
   packed_weight_load_s
   weight_bo_sync_s
   lm_head_bo_sync_s
   scratch_bo_bytes
   weight_bo_bytes
   lm_head_bo_bytes
   ```

7. Only then consider mmap.

   If packed artifact disk load or peak CPU RAM is still visible, replace
   `np.fromfile(...).copy()` with a controlled mmap path and verify that XRT
   upload does not retain an invalid host view.

## Verification

For each step, verify:

```text
python -m compileall src
git diff --check
NPU short prompt: --prompt-len 8 --num-tokens 2
NPU longer prompt: --prompt-len 16 --num-tokens 3
CPU reference token/text match
```

For weight artifact correctness, add static checks:

```text
manifest total bytes == file size
per-layer offsets are monotonic and 64B aligned
each segment shape matches config.weights[name].shape
packed layer slice equals direct pack_llama_layer_weights(config, layer_idx)
```

For external weight BO correctness, first run with one layer or a tiny fused body
if available, then run the full 16-layer decode.

## Success Criteria

The optimization is successful when:

- decode correctness matches the CPU reference for the existing short tests;
- fused scratch no longer contains transformer weights;
- fused scratch no longer contains LM head weights;
- scratch BO size is dominated by packet cache and intermediates, not model
  parameters;
- startup can reuse `llama_iron_packed/weights.bf16.bin`;
- decode uses one fused ELF dispatch per generated token;
- LM head runs inside that same fused ELF from `lm_head_buffer`;
- a future LM head optimization can be developed as an independent path.
