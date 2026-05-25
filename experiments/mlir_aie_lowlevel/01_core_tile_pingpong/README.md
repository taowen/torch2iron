# Experiment 01: Core Tile Ping-Pong

Purpose: verify the explicit MLIR-AIE shape for a compute-tile DMA ping-pong
ring. This is the structure normally hidden behind `ObjectFifo(depth=2)`.

The generated MLIR contains:

- row0 shim tile and one row2 compute tile
- explicit `aie.buffer` ping/pong slots
- explicit `aie.lock`
- `aie.mem` with `aie.dma_start`, `aie.dma_bd`, `aie.use_lock`, and `aie.next_bd`
- runtime shim DMA tasks for coarse DDR input/output movement

Run:

```bash
make
```

Optional compile:

```bash
make compile PYTHON=/var/home/taowen/projects/torch2iron/.venv/bin/python
```

This experiment is a static DMA skeleton, not a full runnable copy kernel.

