# How To Fuse A Transformer Layer In IRON

This note summarizes what the FastFlowLM reverse-engineering says a decode
layer engine needs, what `fast_qwen3/operators/qwen3_layer_fused` currently
provides, and how to keep moving toward a real layer-local fused engine in
IRON.

The short version:

```text
Do not build a full decode layer by stacking logical operators.
Build one layer-specific physical dataflow engine.
Use high-level IRON operators to prove local kernels and numeric boundaries.
Use explicit MLIR-AIE BD/lock/stream contracts for the fused layer schedule.
```

## Evidence Base

FastFlowLM reverse-engineering lives in:

- `/var/home/taowen/projects/MyLM/tools/re/fused-layer-engine/README.md`
- `/var/home/taowen/projects/MyLM/tools/re/fused-layer-engine/10-phase-level-contract.md`
- `/var/home/taowen/projects/MyLM/tools/re/fused-layer-engine/11-complete-dataflow-contract.md`
- `/var/home/taowen/projects/MyLM/tools/re/fused-layer-engine/12-kv-scan-bd-contract.md`

Current IRON-side implementation work lives in:

- `src/models/fast_qwen3/operators/qwen3_layer_fused/op.py`
- `src/models/fast_qwen3/operators/qwen3_layer_fused/design.py`
- `src/models/fast_qwen3/operators/qwen3_layer_fused/static_kv_reader.py`
- `src/models/fast_qwen3/README.md`

The reverse-engineered conclusions below should be read with the same evidence
levels used in the FastFlowLM notes:

- hard: directly supported by transaction dump, CDO/BD decode, or AIE2P disassembly.
- strong: multiple hard facts agree, but exact source names or route labels are missing.
- open: still needs numeric probes, stream-switch decode, or hardware trace.

## What FastFlowLM Needs

FastFlowLM's `layer.xclbin` is not a host transaction that chains
`dequant.xclbin`, `mm.xclbin`, and `attn.xclbin`. It is one transformer-layer
dataflow engine. The host patches descriptors and starts the graph; the layer
then advances through static DMA queues, BD chains, locks, and resident AIE
programs.

### 1. One Layer-Local Physical Schedule

The fused layer needs a physical schedule, not an operator graph:

```text
input/RMS setup
Q projection
K projection
V projection
KV cache write + history read
attention
O projection + residual
FFN norm
up projection
gate projection
SwiGLU
down projection + residual
```

These phases are time-multiplexed over shared physical resources. Fused does
not mean every projection owns a separate tile set.

### 2. Shared Projection Fabric

FastFlowLM reuses the center projection fabric for Q/K/V/O/up/gate/down:

```text
main compute tiles: c2..c5 / r2..r5
main tile count:    16
weight ingress:     row0 shim c2..c5
row1 support:       c2..c5 memtiles
```

For one 512-wide output block:

| Column | Output rows in block | Shim patches | Compute split |
| --- | ---: | --- | --- |
| c2 | 0..127 | patch 0, 1 | r2/r3/r4/r5 each one 32-row chunk |
| c3 | 128..255 | patch 2, 3 | r2/r3/r4/r5 each one 32-row chunk |
| c4 | 256..383 | patch 4, 5 | r2/r3/r4/r5 each one 32-row chunk |
| c5 | 384..511 | patch 6, 7 | r2/r3/r4/r5 each one 32-row chunk |

The exact low-to-high order of r2/r3/r4/r5 is still open, so an implementation
should keep it configurable until numeric probes prove the permutation.

### 3. Online Q4NX MVM, Not Offline Dequant

The projection fabric consumes packed Q4NX weights directly. A patch is:

```text
64 output rows x full K
```

Each main core logically owns:

```text
32 output rows x full K
```

Core-local micro-contract:

```text
for k0 in range(0, K, 256):
    read 256 bf16 scales
    read 256 bf16 zero-points
    read 4096 bytes packed int4 payload
    unpack int4
    dequantize: (q4 - zero_point) * scale
    vector MAC into fp32 accumulator
store 32 bf16 output rows
```

The hot path contains online unpack/dequant/MAC instructions. There is no
evidence of a BF16 weight BO materialized by `dequant.xclbin` for decode.

### 4. Fixed Weight Patch Order

For Qwen3-8B, FastFlowLM's `arg1` weight patch order is:

| Projection | Patch range | Output dim | K | Patch bytes |
| --- | ---: | ---: | ---: | ---: |
| Q | 0..63 | 4096 | 4096 | 0x28000 |
| K | 64..79 | 1024 | 4096 | 0x28000 |
| V | 80..95 | 1024 | 4096 | 0x28000 |
| O | 96..159 | 4096 | 4096 | 0x28000 |
| up | 160..351 | 12288 | 4096 | 0x28000 |
| gate | 352..543 | 12288 | 4096 | 0x28000 |
| down | 544..607 | 4096 | 12288 | 0x78000 |

This matters for IRON because phase fusion is not enough. The generated layer
must also produce the right DDR patch sequence and route those patches to the
right physical columns.

### 5. Replay, Not DDR Round-Trips

FastFlowLM does not send hidden from DDR separately for Q, K, and V. The strong
contract is:

```text
send hidden once
RMSNorm(hidden) lands in row1/local replay buffers
Q/K/V replay that same normalized hidden
O replays attention output
up/gate replay FFN-normalized hidden
down consumes streamed/local SwiGLU product
```

This is the key reason a layer engine cannot be modeled as independent
host-visible operators without losing the performance structure.

### 6. KV Cache Plane Contract

KV cache is four physical planes:

```text
0x000000 -> k03
0x400000 -> v03
0x800000 -> k47
0xc00000 -> v47
```

The semantic order from current writes is `k03`, `k47`, `v03`, `v47`, while
history reads appear in physical-address order. The implementation should keep
the semantic names and physical offsets explicit.

For Qwen3-style 8 KV heads, each plane holds 4 KV heads:

```text
token_stride = 4 heads * 128 head_dim * bf16 = 0x400 bytes
plane_tile   = 16 tokens * 4 heads * 128 dim * bf16 = 0x4000 bytes
```

History read length is rounded to 16-token tiles:

```text
history_len_dwords = ceil(L / 16) * 0x1000
history_len_bytes  = ceil(L / 16) * 0x4000 per plane
```

The tail for L not divisible by 16 must be masked in the attention kernel.

### 7. Static Row1 BD/Lock Rings

The hard KV-scan evidence is a row1 memtile ring, not one runtime DMA task per
KV tile:

```text
ring A:
  bd0  <-> bd1   len=4096, base=0x20000/0x24000, acq64 -> rel65
  bd2  <-> bd3   len=2048, base=0x20000/0x24000, acq65 -> rel66
  bd24 <-> bd25  len=2048, base=0x20080/0x24080, acq66 -> rel64

ring B:
  bd26 <-> bd27  len=4096, base=0x28000/0x2c000, acq67 -> rel68
  bd4  <-> bd5   len=2048, base=0x28000/0x2c000, acq68 -> rel69
  bd28 <-> bd29  len=2048, base=0x28080/0x2c080, acq69 -> rel67
```

This is the strongest argument for leaving high-level ObjectFIFO/Runtime.fill
when implementing the final KV reader. The source may have used an abstraction,
but the binary contract is explicit stream routing plus DMA BD plus lock
acquire/release.

### 8. Runtime Patches Only Dynamic State

A layer run should patch addresses, lengths, and small control values. It
should not generate a fresh runtime DMA graph for each KV tile or projection
chunk.

Runtime responsibilities:

- patch weight BO offsets for the current transformer block.
- patch KV plane offsets and history length for current L.
- patch start/control/RTP fields.
- submit the prepared descriptor queue or runlist.

Static graph responsibilities:

- hold resident compute programs.
- hold memtile/core BD rings.
- hold lock protocol.
- own phase ordering and resource reuse.
- keep intermediates in AIE local memory, streams, or memtiles.

## What `qwen3_layer_fused` Provides Today

`qwen3_layer_fused` currently provides the right implementation shape, but not
yet the full layer.

### 1. An IRON-Compatible Operator Shell

`Qwen3LayerFusedMLIROperator` is still an IRON operator at the outer boundary.
It integrates with the existing artifact and compile flow:

```text
Qwen3LayerFusedMLIROperator
  -> PythonGeneratedMLIRArtifact
  -> DesignGenerator
  -> qwen3_layer_fused(...)
```

The first layer-local slice reuses the current projection implementation to
cover:

```text
RMSNorm
grouped Q/K/V online-Q4 projection into q_current
current K/V persistence into the four-plane cache
```

This is useful because it keeps the public operator boundary compatible while
allowing the internals to become layer-specific.

### 2. Validation Operators Still Matter

The existing high-level operators are still valuable:

- `Q4NXFusedQKVProjection` proves packed-Q4 online MVM and hidden fanout.
- `Q4NXFusedQCurrentProjection` proves grouped q_current generation.
- `QwenCurrentKVPlaneWrite` proves the FastFlowLM-style cache plane write.
- `QwenPlaneAttentionCurrent` proves plane-layout attention semantics.
- `Q4NXFusedLinearResidualProjection` proves post-attention O projection plus residual.

These should be treated as numeric validation boundaries, not final physical
phase boundaries. Once a boundary is validated, fold its semantics into the
layer-specific schedule.

### 3. A Low-Level Static KV Reader Generator

`static_kv_reader.py` is the important shift. It generates text-level MLIR-AIE
for explicit low-level contracts:

- `aie.tile`
- `aie.flow`
- `aie.lock`
- `aie.memtile_dma`
- `aie.dma_bd`
- `aie.next_bd`
- `aie.mem`
- `aiex.npu.writebd`
- `aiex.npu.address_patch`
- `aiex.npu.push_queue`
- `aiex.npu.sync`
- external AIE kernel declarations with `link_with`

This is not normal high-level IRON code. It is an IRON-compatible escape hatch
for the parts of a fused layer where physical control matters.

### 4. Contract Slices Already Proven

The static KV work has been staged in useful slices:

| Slice | What it proves |
| --- | --- |
| `build_static_kv_reader_contract_mlir` | Four-plane descriptor/BD/lock contract can be generated and checked. |
| `build_static_kv_core_dma_contract_mlir` | One plane can enter row1 ring and feed two worker local buffers. |
| `build_static_kv_pair_core_dma_contract_mlir` | K and V can feed worker DMA0/DMA1. |
| `build_static_kv_pair_attention_ingress_contract_mlir` | Two workers can call the real split-K/V attention update ABI. |
| `build_static_kv_dual_pair_attention_ingress_contract_mlir` | Two plane pairs can route and compile for four workers. |
| `build_static_kv_quad_pair_attention_ingress_contract_mlir` | All eight attention workers can route, compile, generate insts/xclbin, and execute on NPU. |
| `build_static_kv_quad_pair_attention_bounded_contract_mlir` | Eight workers can do finite scan, finalize, context egress, and numeric comparison. |
| `build_static_kv_one_pair_two_stage_attention_ingress_contract_mlir` | One plane pair can be read once, forwarded through a source memtile, and split to four workers. |
| `build_static_kv_one_pair_two_stage_attention_bounded_contract_mlir` | That one-read route can run a bounded scan, finalize context, drain four groups, and match CPU reference. |
| `build_static_kv_full_two_stage_attention_bounded_contract_mlir` | Both plane pairs can use one-read two-stage fanout for all eight workers and run with explicit low-level descriptor start. |

The quad-pair ingress currently duplicates each plane-pair read:

```text
mem0/mem2 read k03/v03
mem4/mem6 read k47/v47
```

That is a validation slice, not the final bandwidth-optimal fanout. The
one-pair two-stage slice is the first move toward the real one-read fanout:

```text
mem1 reads k03/v03 once
mem1 forwards full tiles to mem0 and mem2
mem0/mem2 split to g0..g3
```

The full two-stage bounded slice extends this to both plane pairs:

```text
mem1 reads k03/v03 once -> mem0/mem2 -> g0..g3
mem5 reads k47/v47 once -> mem4/mem6 -> g4..g7
```

Its runtime sequence is now explicit low-level NPU control:

```text
writebd        # patch descriptor length/offset/channel fields
address_patch  # bind the descriptor address field to a runtime BO argument
push_queue     # actually start the shim DMA queue
sync           # wait for S2MM context output completion
```

This matters because `writebd` alone is not a start command. It only writes
descriptor fields. A bounded numeric smoke with host-visible output needs
`push_queue` for MM2S/S2MM and `sync` on the output queues.

### 5. Built-In Checkers Are Part Of The Method

Every generator has a corresponding checker that verifies text-level structural
contracts:

- expected number of memtile DMA regions.
- expected worker cores and worker memories.
- expected K/V worker DMA channels.
- expected runtime descriptors or runtime tasks.
- expected ring names and locks.
- expected context egress tasks.
- expected plane offsets and length alignment.

This is not just unit-test decoration. For low-level generated MLIR, structural
checkers are the only way to keep refactors from silently changing the physical
dataflow.

### 6. What Is Still Missing

The current code does not yet provide a complete FastFlowLM-style layer engine.
Missing pieces:

- full projection fabric as a phase-scheduled engine over `c2..c5/r2..r5`.
- shared hidden/RMSNorm replay across Q/K/V.
- O projection consuming attention context without host-visible DDR.
- FFN norm replay across up/gate.
- SwiGLU product feeding down projection locally.
- all-layer phase scheduler with physical tile reuse.
- full one-read `k03/v03` and `k47/v47` fanout for all eight workers.
- production q_current ingress into attention workers without adding a third worker input DMA.
- production context egress into the next projection phase rather than host-visible smoke output.
- final layer runlist across transformer blocks.

## The IRON Fusion Pattern

The practical pattern is:

```text
1. Validate operator semantics in high-level IRON.
2. Identify the physical resources that must be reused.
3. Create a layer-specific operator shell.
4. Generate explicit MLIR-AIE for the static dataflow.
5. Add structural checkers for every physical contract.
6. Add compile-only smoke.
7. Add XRT completion smoke.
8. Add bounded numeric smoke.
9. Replace validation-only duplicate reads with bandwidth-correct fanout.
10. Fold the slice into the layer-local phase schedule.
```

Do not skip from step 1 to a full layer. The resource failures so far show why:
operator-level fusion preserves logical order, but it does not automatically
express physical time-sharing of the same tile, BD, FIFO, and lock resources.

## When High-Level IRON Is Enough

Use high-level IRON when the boundary is local and does not need exact BD/lock
control:

- proving an AIE kernel's math.
- testing a projection microkernel.
- validating q_current layout.
- validating cache plane write semantics.
- validating attention output against CPU.
- prototyping a small graph that does not exhaust dynamic BD resources.

High-level IRON remains the fastest way to discover correct local semantics.

## When To Drop To Explicit MLIR-AIE

Drop to explicit MLIR-AIE when the boundary depends on physical scheduling:

- row1 static BD rings.
- memtile fanout under a fixed output DMA channel budget.
- worker-local ping-pong buffers.
- lock rings that must match producer/consumer backpressure.
- runtime descriptor patching instead of per-tile runtime tasks.
- phase-level reuse of the same physical tiles.
- keeping intermediates off host-visible BOs.

The test is simple: if the question is "which BD, which lock, which DMA
channel, which tile, and which next_bd", high-level operator composition is the
wrong level.

## Recommended Layer Build Plan

### Phase A: Make Attention Static Reader Numerically Closed

Use the bounded attention contracts as the stable test surface:

```text
KV runtime input or writebd-compatible source
row1 static rings
worker DMA0 K / DMA1 V
finite update loop
finalize_bf16
context egress
CPU reference compare
```

This keeps the hard part observable before it is hidden inside a larger layer.

### Phase B: Remove Duplicate Plane Reads

Move from the quad duplicate baseline to one-read fanout:

```text
k03/v03 source memtile -> split memtiles -> g0..g3
k47/v47 source memtile -> split memtiles -> g4..g7
```

The goal is not simply fewer runtime tasks. The goal is matching the
FastFlowLM contract: one coarse plane reader, static row1 fanout/split, worker
streams driven by locks.

### Phase C: Add q_current Ingress

Do not give each worker a third input DMA channel. The current attention ABI is
already K on DMA0 and V on DMA1. Candidate strategies:

- compute q_current in the same worker before history scan.
- preload q_current into worker local memory through an earlier phase.
- broadcast q_current through a separate phase and then reuse local storage.

The first production implementation should prefer correctness and resource
stability over perfect overlap.

### Phase D: Turn Context Egress Into O-Projection Input

The bounded smoke drains context to host for verification. The layer engine
should instead route context into the O projection phase:

```text
attention context
  -> local/memtile replay
  -> c2..c5/r2..r5 Q4NX O projection
  -> residual update
```

Do not keep the host-visible context BO as a permanent boundary.

### Phase E: Build The Projection Phase Scheduler

The projection scheduler should be layer-specific:

```text
for phase in [Q, K, V, O, up, gate, down]:
    choose input replay source
    patch/queue phase weight patches
    arm shared projection fabric
    let BD/locks drive all output blocks
```

The main fabric mapping should come from one source of truth, currently
`phase_tiles.py`, and should keep row-order permutations configurable until
verified.

### Phase F: Integrate Runtime Patching

Runtime should patch:

- layer weight offsets.
- KV plane base offsets.
- current token offset.
- rounded history length.
- context length / mask controls.
- start or phase-control fields.

Runtime should not emit one DMA task per KV chunk in the final decode path.

## Coding Guidelines For This Style

Keep the low-level contract declarative:

```python
@dataclass(frozen=True)
class RingDescriptor:
    name: str
    mlir_locks: tuple[int, int, int]
    load_channel: int
    load_bds: tuple[int, int]
    half0_channel: int
    half0_bds: tuple[int, int]
    half1_channel: int
    half1_bds: tuple[int, int]
```

Generate MLIR from descriptors, not from scattered string fragments. Use small
helpers for:

- lock declarations.
- ring buffers.
- load rings.
- split/forward rings.
- worker DMA rings.
- runtime descriptors.
- context egress.

Every generator should have a checker. The checker should enforce physical
facts, not just "the string contains a function name".

Every new slice should have three scripts or commands:

```text
contract smoke: generate and structurally check MLIR
compile smoke: run aiecc through routing/link/xclbin
XRT smoke: run the generated xclbin/insts on NPU
```

Whenever possible, add a bounded numeric smoke before calling the slice
implemented.

## Common Pitfalls

Do not mistake compile success for dataflow correctness. A route can compile
while the half-buffer split is only a structural placeholder. Numeric or
pattern-based probes are needed to prove layout.

Do not treat duplicate reads as final fanout. Duplicate reads are acceptable for
routing baselines and bounded numeric tests; they are not the bandwidth model to
ship.

Do not make every logical operator own its own FIFO/BD/worker set. A fused
layer needs physical resource reuse across phases.

Do not force attention to have three worker input streams. The proven worker
shape is K on one DMA input and V on the other; q_current must be staged or
computed without exceeding input DMA limits.

Do not preserve host-visible intermediate BOs after validation. They are useful
for tests, but the fused layer should keep Q/K/V, attention context, FFN
intermediates, and SwiGLU products local.

## Is This "The Future IRON Style"?

Not as raw text MLIR. Raw string-generated MLIR is a necessary bridge, not the
desired long-term user API.

The future IRON abstraction for this class of model should look like the
semantics now being hand-written:

```text
PlacedTile
StaticBDRing
RuntimePatchedDescriptor
LockProtocol
MemtileFanout
WorkerLocalPingPong
LayerPhase
LayerRunlist
```

The important shift is already clear: full decode-layer fusion needs an
explicit physical dataflow API. Standard operator-level fusion and runlist
submission are still useful, but they are not enough to express FastFlowLM's
layer engine by themselves.

Until IRON has those first-class abstractions, the practical route is:

```text
IRON-compatible operator shell
  + layer-specific Python MLIR generator
  + explicit MLIR-AIE BD/lock/stream contracts
  + structural checkers
  + bounded numeric smokes
```

That is the pattern `qwen3_layer_fused` is establishing.
