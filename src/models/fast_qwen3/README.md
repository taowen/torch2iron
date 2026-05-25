# Fast Qwen3

This directory is the new high-performance Qwen3 path.  It is intentionally
separate from `quantized_qwen3` so the fused-layer work can move without adding
runtime switches to the old implementation.

## Direction

The target decode architecture is:

- one fused layer engine per transformer layer;
- packed Q4 weights consumed directly by the layer kernel;
- RMSNorm output replayed through Q, K, and V projection phases;
- FFN norm output replayed through up and gate projection phases;
- KV cache stored in the layout attention consumes directly;
- runlist submission across layers once the layer engine is validated.

The first checked-in milestone is the Q4NX-style artifact plus AIE operator
boundaries that validate packed Q4NX online-MVM, hidden fanout, current-aware
attention, and attention-output projection.  These operators are validation
boundaries; the final layer engine should use the phase schedule documented from
FastFlowLM rather than preserving every temporary operator boundary.

## Artifact Layout

The fast artifact lives under `fast_qwen3_q4nx/` and contains:

- `manifest.json`
- `weights.q4nx.bin`

Each Q4NX chunk covers `32 output rows x 256 input cols` and stores:

- 256 bf16 scales;
- 256 bf16 zero-points;
- 4096 bytes of packed uint4 weights.

The dequant formula is:

```text
weight = (q4 - zero_point) * scale
```

The host-visible patch unit is `64 output rows x full K`, ordered by K chunk
inside each output shard.  This matches the layer-engine direction rather than
the old generic W4A16 GEMM tile layout.

## First AIE Boundary

`Q4NXFusedQKVProjection` consumes:

- one bf16 raw hidden vector;
- one bf16 RMSNorm weight vector;
- one contiguous Q/K/V patch stream per output patch, ordered by K chunk as
  `Q,K,V`;
- one contiguous bf16 output group, ordered by output patch and then Q/K/V.

This validates the online-Q4 MVM and hidden fanout mechanics.  FastFlowLM's
layer contract uses Q, K, and V as sequential phases over the same hidden-norm
replay, so this interleaved QKV operator is not the final phase schedule.  The
current kernel computes normed hidden once, forwards it to eight QKV workers,
uses AIE vector `uint4 -> bf16 -> mac/reduce` operations, and keeps the Q4NX
zero-point semantics in the online dequant path.  The default operator shape
computes eight 64-row output patches on eight compute columns.

`Q4NXFusedUpGateProjection` uses the same normed-hidden fanout pattern for the
FFN projection pair.  Its packed stream is ordered by output patch and K chunk as
`up,gate`.  This validates shared FFN-norm fanout; the final layer contract uses
up and gate as sequential phases sharing the same FFN-norm replay.

`Q4NXFusedQCurrentProjection` is the group-local Q/K/V projection shape needed
by decode attention.  Instead of producing a generic `[patch, Q/K/V, 64]`
surface, it consumes a group-major packed Q4NX stream and writes
`q_current = [Q heads, current K, current V]` directly for one or more KV
groups.  The kernel computes RMSNorm once, then reuses the normed hidden stream
while eight output-patch workers walk each KV group.  For each Qwen3-0.6B group
this is eight 64-row patches: four Q patches, two K patches, and two V patches,
which fits the current eight-column worker layout without computing K/V patches
for groups the following attention worker will not consume.

`QwenChunkedAttentionCurrent` is the first layer-local decode attention boundary.
It receives Q plus the current K/V as one per-group stream, scans the packet
cache directly, and uses the current K/V row inside the current chunk without
materializing an updated packet chunk.  This keeps each worker to two input
streams (`q_current` and packet), which is required by AIE input DMA limits.
`QwenCurrentKVCacheWrite` persists that same current K/V row as a separate
small-write data movement step.  Trying to write an updated full packet chunk
from the attention worker exceeds tile-local SRAM, so the current working
boundary is attention-current plus cache-writer in the same fused ELF.
Trying to fold the small write into this attention operator exposed the next
real constraint: an extra update output in the attention worker corrupts the
context result, while a separate row3 writer worker fed by the same `q_current`
FIFO works for one group but exceeds `SequentialPlacer`'s 8-group endpoint
budget.  The fused-layer implementation therefore needs explicit placement for
the update stream instead of another automatic runlist composition.

`QwenCurrentKVPlaneWrite` is the first FastFlowLM-layout cache writer.  It
writes the current K/V rows from grouped `q_current` into the four physical KV
planes ordered as `k03, v03, k47, v47`, with each token row stored as
`4 KV heads x head_dim`.  It now treats the plane as an inout/external BO and
only drains the current K/V rows, preserving all other history rows in-place.
The goal is to replace the old packet-cache writer/reader with a plane layout
that attention can consume directly.

`QwenPlaneAttentionCurrent` is the matching plane-layout attention reader.  It
streams each physical K/V tile through two plane-pair readers: `k03/v03` serves
groups 0..3 and `k47/v47` serves groups 4..7.  The runtime fill uses a
group-major tap, so row1 can split each coarse DDR read into four contiguous
group streams instead of rereading the same plane pair once per group.  The
current slot still bypasses the plane and uses the current K/V carried in
`q_current`, which lets the layer engine write cache and compute attention
without reading back the just-written row.  `attend_seq_len` remains the actual
valid-token count, so the last tile's tail rows are masked inside the kernel.
The high-level IRON path now supports 64 tokens with 16-token tiles and 128
tokens with 32-token tiles.  A 128-token path with 16-token tiles still exceeds
dynamic BD budget, so a final FastFlowLM-style reader should replace those
per-chunk runtime fills with a static row1 memtile BD chain and runtime-patched
descriptors.

The stable decode-attention validation boundary is now
`Q4NXFusedQCurrentProjection -> QwenPlaneAttentionCurrent`.  This keeps packed
Q4 projection and plane-layout attention in the same fused ELF and validates the
replacement for the older packet-cache attention path.  The writer is still a
small dataflow boundary: `QwenCurrentKVPlaneWrite -> QwenPlaneAttentionCurrent`
runs in one high-level IRON fused ELF and proves a row written into the
FastFlowLM-style plane cache can be consumed as history by the next-token
attention step.  This path uses `kv_plane` as a persistent external BO and the
smoke checks the whole plane remains equal to the CPU in-place reference after
the fused call.

The stronger high-level boundary is now
`Q4NXFusedQCurrentProjection -> QwenCurrentKVPlaneWrite ->
Q4NXFusedQCurrentProjection -> QwenPlaneAttentionCurrent`.  It models two decode
steps inside one fused ELF: the first packed-Q4 projection writes current K/V
into the persistent plane cache, the second packed-Q4 projection produces the
next token's q_current, and plane attention reads the just-written history row.
This shows the cache update path can stay in high-level IRON for now; the
remaining layer-engine work is placement/phase scheduling, not proving that
current-row persistence requires low-level CDO.

`QwenQKVToQCurrent` bridges the QKV projection patch layout to the grouped
attention layout.  With the current 8-patch QKV projection it assembles the
first two KV groups (`4 Q patches + 2 K patches + 2 V patches` per group) into
`q_current` inside the fused ELF, so Python no longer performs that format
conversion for the integration boundary.

`Q4NXFusedLinearProjection` is the current `o_proj` boundary after attention.
Qwen3-0.6B `o_proj` has `K=2048`; a single worker consuming all eight 256-wide K
chunks for 64 output rows times out at runtime.  The current operator follows
the FastFlowLM phase-level contract instead: one 512-wide output block maps to
the c2..c5/r2..r5 projection fabric.  Each 64-row output patch is split into two
32-row Q4NX chunk workers, and each worker consumes its own full-K packed Q4
buffer in-core.  There is no cross-tile K reduce.  The default operator covers
one eight-patch `o_proj` block using sixteen projection workers; full Qwen3-0.6B
`o_proj` has two such blocks.  To keep shim DMA within limits, activation enters
from c0 once and broadcasts to all projection workers, while c2..c5 carry only
the two weight patch ingress streams per projection column.  Weight is split per
64-row patch on row1 and output is joined per 64-row patch before draining.  The
stable integration boundary today is
`QwenChunkedAttentionCurrent -> Q4NXFusedLinearProjection`.  Directly stacking
`Q4NXFusedQCurrentProjection -> QwenChunkedAttentionCurrent ->
Q4NXFusedLinearProjection` in the current automatic fused runlist times out or
produces incorrect O output, so the full layer path needs a manually designed
layer-local dataflow rather than more temporary operator chaining.

`Q4NXFusedLinearResidualProjection` is the first residual-path phase boundary.
It keeps the same 512-wide `o_proj` fabric and adds a second stage on edge
compute tiles that consumes each 64-row projected patch plus a residual block and
writes the residual-updated patch.  This validates the next layer-local step:
projection output can flow into residual math inside the same ELF without
returning to Python or materializing a separate host-side add.  The current
integration boundary `QwenChunkedAttentionCurrent ->
Q4NXFusedLinearResidualProjection` now covers `attention -> O projection ->
residual` for one or more 512-wide blocks.  With `--o-block-count 2`, it covers
the full Qwen3-0.6B hidden-size output surface (`[16, 64]`).

The stable direct boundary today is
`Q4NXFusedQCurrentProjection -> QwenChunkedAttentionCurrent`.  The
`QwenCurrentKVCacheWrite` path is covered by the attention-current smoke; a
three-op direct projection + writer + attention runlist currently executes once
but is not re-entrant, so the real fix is a manually placed layer-internal cache
write path rather than preserving this temporary three-op boundary.

## Smoke

```bash
scripts/run_fast_qwen3_qkv_smoke.sh --repack
scripts/run_fast_qwen3_qkv_operator_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_q_current_operator_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_q_current_operator_smoke.sh --num-kv-groups 8 --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_q_current_attention_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_q_current_attention_smoke.sh --num-kv-groups 8 --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_up_gate_operator_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_o_projection_operator_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_o_projection_residual_operator_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_attention_o_projection_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_attention_o_projection_residual_smoke.sh --o-block-count 2 --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_kv_plane_write_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_kv_plane_write_attention_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_kv_plane_write_attention_smoke.sh --attend-seq-len 7 --write-slot 5 --attention-slot 6 --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_plane_attention_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_plane_attention_smoke.sh --attend-seq-len 128 --current-slot 127 --tile-size 32 --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_q_current_plane_attention_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_q_current_plane_attention_smoke.sh --attend-seq-len 128 --current-slot 127 --tile-size 32 --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_q_current_plane_write_attention_smoke.sh --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_q_current_plane_write_attention_smoke.sh --attend-seq-len 7 --write-slot 5 --attention-slot 6 --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_attention_current_smoke.sh --num-kv-groups 8 --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_qkv_attention_current_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_qkv_operator_smoke.sh --trace-size 131072 --timed-iters 1
```

The smoke check packs a model if needed, runs layer-0 fused Q/K/V reference on a
random hidden row, and prints output shapes plus basic value stats.  The
operator smoke additionally compiles the first Q4NX fused RMSNorm+QKV patch as a
full ELF, runs it on the NPU, compares it with the CPU patch reference, prints
wall-time samples, and can emit raw event trace data.
