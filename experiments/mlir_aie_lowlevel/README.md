# MLIR-AIE Low-Level Experiments

These experiments isolate the low-level pieces needed to reproduce a
FastFlowLM-style fused layer dataflow without relying on high-level
`Runtime.fill` expansion.

Each experiment directory is self-contained: it has its own generator,
checker, Makefile, and README. The default `make` target only generates MLIR
and runs static checks. The optional `make compile` target invokes `aiecc` when
the local MLIR-AIE/Peano toolchain is available.

Experiments:

1. `01_core_tile_pingpong`:
   Explicit compute-tile `aie.mem` ping-pong DMA ring with locks and BD next
   pointers.
2. `02_memtile_kv_plane_ring`:
   Explicit row1 memtile KV-plane ring contract with FastFlowLM-like BD IDs,
   lock IDs, and 4096/2048-word stages.
3. `03_runtime_writebd_patch`:
   Explicit `aiex.npu.writebd` runtime descriptor patch sequence for four KV
   planes.

Run all static checks:

```bash
for d in 01_core_tile_pingpong 02_memtile_kv_plane_ring 03_runtime_writebd_patch; do
  make -C "experiments/mlir_aie_lowlevel/$d"
done
```

