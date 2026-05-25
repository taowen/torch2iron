# Experiment 02: Memtile KV Plane Ring

Purpose: model the row1 memtile KV-history reader as explicit static BD and
lock rings, instead of generating one host DMA task per KV chunk.

The generated MLIR focuses on the FastFlowLM-style contract:

- two independent rings, A and B
- FastFlowLM reference lock IDs `64,65,66` and `67,68,69` are preserved in
  comments
- MLIR-AIE lock IDs use compiler-legal local IDs `4,5,6` and `7,8,9`
- 4096-word full-plane stage
- two 2048-word split stages
- even/odd ping-pong BDs:
  - ring A: `0/1`, `2/3`, `24/25`
  - ring B: `26/27`, `4/5`, `28/29`

Run:

```bash
make
```

Optional compile:

```bash
make compile PYTHON=/var/home/taowen/projects/torch2iron/.venv/bin/python
```

This is a low-level dataflow skeleton. It intentionally does not include the
attention worker implementation.

Note: `aie.lock` verifier currently rejects lock IDs above 63, so the generator
keeps FastFlow reference IDs as annotations and emits legal local lock IDs for
compilation.
