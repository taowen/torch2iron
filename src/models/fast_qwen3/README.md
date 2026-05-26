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
The final implementation entry is `Qwen3LayerFusedMLIROperator`: a
Qwen3-specific fused layer operator that generates one layer-local `aie.device`
instead of stitching child operator `runtime_sequence`s together.  The first
checked-in slice of that operator covers q_current projection plus current K/V
plane persistence; subsequent work should extend this layer-local design with
the static KV reader and attention/O/FFN phases.
The shared placement contract for the current projection fabric lives in
`phase_tiles.py`; validation operators use this single source of truth for the
c2..c5/r2..r5 Q4NX projection workers and the edge residual workers.

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
`Qwen3LayerFusedMLIROperator` is the first layer-local variant of this phase:
after producing `q_current`, the same layer-specific device copies current K/V
rows into the persistent four-plane cache.  This removes one standalone writer
`RunOp` from the projection/write boundary while preserving the external
`kv_plane` BO in-place.  It intentionally starts as a narrow slice; it should be
extended into the complete layer engine instead of adding another automatic
multi-operator runlist boundary.

The next layer-local boundary is the static KV reader contract in
`operators/qwen3_layer_fused/static_kv_reader.py`.  It brings the FastFlowLM
row1 memtile ring and runtime shim BD patching into the `fast_qwen3` path:
ring A uses `0/1,2/3,24/25`, ring B uses `26/27,4/5,28/29`, and runtime patches
the four physical planes `k03,v03,k47,v47` with `ceil(L/16) * 0x1000` dwords.
The contract also fixes the worker-facing stream mapping: groups `g0..g3`
consume `k03/v03` on workers `c0r2..c3r2`, and groups `g4..g7` consume
`k47/v47` on workers `c4r2..c7r2`.  Each worker now declares the real
`llama_chunked_attention_init_f32 -> qwen_plane_group_attention_update_bf16 ->
llama_chunked_attention_finalize_bf16` ABI over local q_current, plane tile,
state, accumulator, and output buffers.  The generated contract MLIR parses,
routes, links the current attention object, and compiles with `aiecc`.  The
q/kv/context FIFO route contract is also emitted and checked, but deliberately
kept as BD/lock route metadata instead of high-level `aie.objectfifo`: a direct
ObjectFIFO version over `c0r1` fails resource allocation with "number of output
DMA channel exceeded".  The next implementation step is replacing the local ABI
buffers with explicit static ring stream reads rather than ordinary per-chunk
`Runtime.fill`.

`qwen3_layer_static_kv_core_dma_contract_smoke.py` is the first executable slice
of that lower-level route.  It narrows the scope to one physical plane
(`k03`) and one memtile ring (`ring_a`), then connects `ring_a` half0/half1 to
`worker_g0/worker_g1` local ping-pong buffers through core-tile S2MM DMA and
worker-local locks.  This contract compiles with `aiecc`, so the next step is
expanding the same explicit BD/lock pattern from two half consumers to the
attention worker layout instead of reintroducing high-level KV ObjectFIFO
fanout.

`qwen3_layer_static_kv_pair_core_dma_contract_smoke.py` is the next slice: it
uses `ring_a` for `k03` and `ring_b` for `v03`, then sends each worker both K
and V through the two core S2MM input channels.  This also compiles with
`aiecc`, which confirms the layer-local static reader can feed a worker with
the two streams the current attention ABI needs.  The remaining work is to
replace the consumer no-op loop with the attention update kernel and then scale
the same route to the second plane pair.

`qwen3_layer_static_kv_pair_attention_ingress_contract_smoke.py` replaces that
consumer no-op with the real split-K/V attention update ABI.  The static reader
still narrows scope to `k03/v03` and two workers, but each worker now calls
`qwen_plane_group_attention_update_split_bf16` on ping and pong K/V buffers.  It
compiles with `aiecc` after rebuilding the plane attention kernel object via
`scripts/build_fast_qwen3_plane_attention_kernel_object.sh`.

`qwen3_layer_static_kv_dual_pair_attention_ingress_contract_smoke.py` scales
that ingress one step further without returning to per-group `Runtime.fill`:
`mem0` reads `k03/v03` from shim column 0 and feeds `worker_g0/g1`, while
`mem4` reads `k47/v47` from shim column 4 and feeds `worker_g4/g5`.  The
generated MLIR still uses only four runtime `writebd` descriptors for the four
physical planes, and compiles through routing, four Peano worker cores, external
attention object link, and xclbin generation.  This validates the dual
plane-pair static ingress shape.  The XRT smoke also executes this xclbin on
NPU and returns `ERT_CMD_STATE_COMPLETED` with a sub-millisecond sample, so this
slice is not compile-only.  The remaining attention work is filling in the other
four group workers and adding q_current ingress/context egress.

`qwen3_layer_static_kv_quad_pair_attention_ingress_contract_smoke.py` fills in
all eight attention workers with the same static-reader shape: `mem0/mem2` both
read `k03/v03` and feed `g0/g1` plus `g2/g3`, while `mem4/mem6` both read
`k47/v47` and feed `g4/g5` plus `g6/g7`.  This is intentionally still a
validation slice: it duplicates each plane-pair read once, so it is not the
final bandwidth-optimal FastFlowLM fanout.  It does prove the full 8-worker
split-K/V attention ingress can route, compile, generate NPU insts, and execute
on NPU with `ERT_CMD_STATE_COMPLETED`.  The next step is removing the duplicate
plane-pair read by decoding a real static split/fanout route, then wiring
q_current ingress and context egress.

`qwen3_layer_static_kv_one_pair_two_stage_attention_ingress_contract_smoke.py`
is the first non-duplicating fanout slice.  It covers only one plane pair
(`k03/v03 -> g0..g3`) but reads each plane from DDR once: source `mem1` receives
the full K/V tiles from shim column 1, forwards full-tile streams to `mem0` and
`mem2`, then those two second-stage memtiles split K/V halves to `g0/g1` and
`g2/g3`.  This avoids asking one memtile to both feed local workers and fan out
to another pair, which would exceed the four output-DMA channel budget.  The
contract routes, compiles, generates xclbin/insts, and executes on NPU with
`ERT_CMD_STATE_COMPLETED`.  The next step is to turn this one-pair ingress into
a bounded/finalize/context slice, then instantiate the same pattern for
`k47/v47`.

`qwen3_layer_static_kv_one_pair_two_stage_attention_bounded_contract_smoke.py`
adds the numeric closure for that one-read slice.  It keeps the same
`k03/v03 -> g0..g3` two-stage fanout, changes the worker loop to a bounded
history scan, finalizes BF16 context, and drains four context groups to the
host.  Like the bounded quad baseline, it uses two coarse runtime MM2S KV
input tasks so the host has a real completion boundary for context output;
`q_current` is locally zeroed on each worker and does not consume a third input
DMA.  The XRT smoke compiles and executes this slice on NPU and compares
against the exact contract CPU reference (`max_abs_error=0.0087890625` in the
current run).

`qwen3_layer_static_kv_full_two_stage_attention_bounded_contract_smoke.py`
extends that same one-read structure to all eight workers.  `mem1` reads
`k03/v03` once and fans out to `mem0/mem2`, while `mem5` reads `k47/v47` once
and fans out to `mem4/mem6`; those four second-stage memtiles split K/V halves
to `g0..g7`.  This removes the duplicate plane-pair DDR read from the quad
baseline while preserving the bounded finite-loop/finalize/context-drain smoke
surface.  Its runtime sequence now spells out the patched descriptor start path
directly: `aiex.npu.writebd`, `aiex.npu.address_patch`, `aiex.npu.push_queue`,
and S2MM `aiex.npu.sync` for host-visible context completion.  The XRT smoke
executes and matches the CPU contract reference (`max_abs_error=0.0087890625`,
current sample about 2.01 ms).  It is a correctness/routing milestone, not yet
the final fastest shape: the two-stage fanout has more static fabric than the
duplicate-read baseline, so the next optimization is reducing that fabric cost
and moving context from host egress into the O-projection phase.

`qwen3_layer_static_kv_quad_pair_attention_bounded_contract_smoke.py` keeps
that duplicate-read quad shape as a stable routing baseline, but changes the
worker program from an infinite ingress loop into a finite history scan followed
by `llama_chunked_attention_finalize_bf16` and context drain.  For the numeric
smoke it feeds the row1 rings with eight coarse runtime MM2S tasks instead of
`npu.writebd`, because `writebd` only patches descriptors and is not a useful
completion boundary for host-visible context output.  `q_current` is
deterministically zeroed on tile, avoiding a third worker input DMA.  The XRT
smoke compiles the 8-worker bounded slice, executes it on NPU, drains all eight
context groups, and compares against a CPU reference for the exact duplicate
read contract (`max_abs_error=0.0087890625` in the current run).

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
The first `Qwen3LayerFusedMLIROperator` slice works as an isolated phase, but
replacing the first projection+writer pair in this multi-step automatic runlist
with any merged projection/write device currently times out.  That reinforces
the same boundary: use validation operators to prove local phase semantics, then
extend the layer-local schedule instead of adding more automatic runs.

`QwenPlaneAttentionCurrent -> Q4NXFusedLinearResidualProjection` is the matching
post-attention phase boundary.  It validates that four-plane attention output can
feed the packed-Q4 O projection plus residual add in the same fused ELF.  The
automatic runlist boundary does not scale indefinitely: extending one fused call
through two q_current projections, persistent plane write, plane attention, and
O residual times out even with a one-patch O projection.  That is the current
line where the work should switch from stacking validation operators to a
single hand-placed layer phase schedule.

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
`Q4NXFusedLinearProjection` and `Q4NXFusedLinearResidualProjection` now share the
same `phase_tiles.py` placement contract, so future layer-local schedules do not
duplicate the projection fabric mapping in each design.

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
scripts/run_fast_qwen3_q_current_projection_plane_write_smoke.sh --warmup-iters 1 --timed-iters 3
scripts/run_fast_qwen3_layer_static_kv_contract_smoke.sh --attend-seq-len 128 --tile-size 16
scripts/run_fast_qwen3_layer_static_kv_core_dma_contract_smoke.sh --attend-seq-len 128 --tile-size 16
scripts/run_fast_qwen3_layer_static_kv_pair_core_dma_contract_smoke.sh --attend-seq-len 128 --tile-size 16
scripts/build_fast_qwen3_plane_attention_kernel_object.sh
scripts/run_fast_qwen3_layer_static_kv_pair_attention_ingress_contract_smoke.sh --attend-seq-len 128 --tile-size 16
scripts/run_fast_qwen3_layer_static_kv_dual_pair_attention_ingress_contract_smoke.sh --attend-seq-len 128 --tile-size 16
scripts/run_fast_qwen3_layer_static_kv_dual_pair_attention_ingress_xrt_smoke.sh
scripts/run_fast_qwen3_layer_static_kv_quad_pair_attention_ingress_contract_smoke.sh --attend-seq-len 128 --tile-size 16
scripts/run_fast_qwen3_layer_static_kv_quad_pair_attention_ingress_xrt_smoke.sh
scripts/run_fast_qwen3_q_current_plane_write_attention_smoke.sh --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_q_current_plane_write_attention_smoke.sh --attend-seq-len 7 --write-slot 5 --attention-slot 6 --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_plane_attention_o_projection_residual_smoke.sh --warmup-iters 1 --timed-iters 1
scripts/run_fast_qwen3_attention_current_smoke.sh --num-kv-groups 8 --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_qkv_attention_current_smoke.sh --warmup-iters 1 --timed-iters 5
scripts/run_fast_qwen3_qkv_operator_smoke.sh --trace-size 131072 --timed-iters 1
```

The smoke check packs a model if needed, runs layer-0 fused Q/K/V reference on a
random hidden row, and prints output shapes plus basic value stats.  The
operator smoke additionally compiles the first Q4NX fused RMSNorm+QKV patch as a
full ELF, runs it on the NPU, compares it with the CPU patch reference, prints
wall-time samples, and can emit raw event trace data.
