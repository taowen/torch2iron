# Experiment 03: Runtime `npu.writebd` Patch

Purpose: isolate the runtime side of a FastFlowLM-style decode dispatch:
runtime patches descriptor fields directly instead of emitting one high-level
DMA task for every logical KV chunk.

The generated MLIR contains one `aie.runtime_sequence` with four
`aiex.npu.writebd` operations for the physical KV planes:

- `k03` at `0x000000`
- `v03` at `0x400000`
- `k47` at `0x800000`
- `v47` at `0xc00000`

The default history length is one 16-token tile per plane:

```text
buffer_length = 0x1000 dwords
```

The emitted shim BD IDs are compiler-legal local IDs `12..15`. Earlier
FastFlowLM traces often patch high BD slots such as `15`; MLIR-AIE verifies BD
IDs per shim DMA local namespace, so this experiment keeps the important part:
direct runtime patching of `buffer_offset`, `buffer_length`, packet fields, and
`use_next_bd=0`.

Run:

```bash
make
```

Optional compile:

```bash
make compile PYTHON=/var/home/taowen/projects/torch2iron/.venv/bin/python
```
