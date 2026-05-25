# Optimization Plan

## 目标

`quantized_qwen3` 和 `exported_llama3` 的正式代码必须收敛到一套面向 AMD NPU 的最高性能执行形态。batch、context variant 和参数选择只能服务最终 dataflow，不能替代 microkernel、内存布局和 tile pipeline 的高性能实现：

- 权重以压缩 W4 格式从磁盘进入 AIE kernel。
- 反量化、矩阵微内核、激活和必要的 reduction 在同一条 tile dataflow 中完成。
- decode 单 token 最终只提交一个包含 transformer 和 `lm_head` 的 ELF。
- prefill 按 chunk 执行，每个 chunk 走 ELF，最后 chunk 直接产生 logits。
- attention/KV cache 使用 streaming tile，不做整块 KV copy，不做 cache 格式往返转换。
- batch=1 也通过 N/head/FFN shard 并行充分使用 column/core，而不是只依赖 batch 维度。

所有进入正式代码的改动都必须服务这套形态。文档只描述最终主线；不能成为最终主线的代码不保留。

## 最终最高性能写法

### 1. 单一离线权重格式

离线 artifact 只保存推理 kernel 直接消费的格式：

- W4A16 group quant 权重按 K 维 group 保存，group size 与 GEMM `tile_k` 对齐。
- signed int4 权重和 bf16 scale 按 AIE `aie::mmul<4,8,8>` 的 `s x t` subtile 顺序排布。
- N 维按 column/core shard 连续存储，避免运行时重排。
- `lm_head`、attention projection、FFN projection 使用同一类 packed W4 tile 格式。
- `lm_head` 的 N/vocab 维度按 8 column × 4 core row 对齐，允许 vocab shard 使用全部 32 个 AIE core。
- 不保存预展开 bf16 GEMM 权重。
- 不维护多套运行时格式选择。

磁盘格式必须让 host 只做 mmap/load，不做 reshape、repeat、transpose 或 dequant。

### 2. W4A16 fused dequant GEMM

正式 GEMM kernel 的结构固定为：

1. 从 DDR 读取 packed int4 B tile 和对应 scale。
2. 在 AIE tile 内向量化 unpack int4。
3. 直接生成 `MMUL::size_B` bf16 vector。
4. 立即喂给 `aie::mmul` 累加。
5. 只把最终 C tile 写回。

不允许把 int4 展开成完整 bf16 权重 tensor 再交给另一个 GEMM。`dequant` 不是独立 op，而是 GEMM microkernel 的一部分。

GEMM 并行方式：

- M 维负责 batch/prompt rows。
- N 维负责输出通道 shard。
- K 维按 group/tile 流式推进。
- activation A tile 需要在多个 N shard 间复用，不能为每个 N shard 重复从 DDR 拉同一份 A。
- 长 K 的 decode projection 不能每个 128-K tile 都单独 load/store C。Q、attention output、FFN down 这类非 paired projection 使用 K-group fused accumulation：一个 worker call 消费多个 K tile，在 tile-local accumulator 中完成跨 K group 累加，最后只 store 一次 C tile。
- FFN gate/up、attention key/value 这类共享 activation 的投影使用 paired K-group stream fusion：一个 activation stream，一个 paired packed weight stream，两个 output stream；worker call 消费多个 K tile，在 tile-local accumulator 中完成跨 K group 累加，最后只 store 一次 C tile。
- decode batch=1/2 时不能用 padding M rows 伪造并行度。正式 GEMM 需要把 N shard 同时铺到 column 和 row，多个 core row 处理同一个 M tile 的不同输出通道；A tile 通过 broadcast/forward 复用，不能为每个 N-row shard 重新从 DDR 读取。
- N shard 的 runtime DMA 也必须按 column 聚合：QP 不能为每个 row×column 建独立 DDR input，C 不能为每个 row×column 建独立 drain。QP 需要 column-level DMA 后在 mem tile split 到 row shard；C 需要 row shard 在 mem tile join 后用 strided column-level drain 写回。否则 shim channel 会先于 compute 成为瓶颈。

decode 的 K/V 和 FFN gate/up 正式实现使用 `W4A16PairedKGroupGEMM`：

- packed artifact v13 在 `paired_linears` 中保存 K/V 和 gate/up 的 combined W4 tile。
- pair 维度在磁盘格式中位于 `k_tile` 和 `n_block` 之间，kernel 每个 K group 一次读取多组 paired packed B tile。
- worker 只消费 activation A 和 paired QP 两个 input FIFO；两个输出分别写回 K/V 或 gate/up。
- 不在 host 热路径拼接、reshape 或复制 paired weight。

### 3. Decode 单 token 一个 ELF

decode 的最终提交单元是一个 ELF：

- 输入：当前 token embedding、position/rope 信息、packet KV cache、packed W4 weights。
- 内部：所有 transformer layers、final norm、`lm_head`。
- 输出：logits、追加后的 present K/V。

这意味着 `lm_head` 不是独立 dispatch。它必须使用同一套 W4A16 fused dequant GEMM，并按 vocab/N shard 并行。
离线 packed artifact 必须把 `lm_head` padding 到 4-row N-shard 可整分的长度，运行时只在 logits 读取阶段裁回真实 vocab。

context window 使用多个 decode ELF variant 解决静态 shape 与性能之间的冲突：

- 每个 variant 有固定最大 context。
- 运行时选择足够覆盖当前 context 的最小 variant。
- variant 之间共享同一套权重 artifact 和 kernel 逻辑。
- decode kernel 只读取有效 KV tile，不能因为大 context 上限固定扫描全部空位。

### 4. Prefill chunk ELF

prefill 的最终形态是 chunked ELF：

- prompt 按固定 chunk size 进入 prefill ELF。
- 每个 chunk 在 ELF 内完成 transformer。
- 每个 chunk 只把新增 K/V 追加到 packet cache。
- 最后 chunk 在同一个 ELF 内执行 final norm 和 `lm_head`，直接输出 logits。
- chunk size 由 AIE local memory、attention tile、weight stream 和 event trace 决定，不用手工调参作为性能方案。

prefill 与 decode 使用相同的权重格式、同类 GEMM microkernel 和同一套生成组织。

### 5. Attention/KV streaming

attention 的正式写法必须按 KV tile 流式运行：

- packet cache 的物理 layout 与 attention 读取顺序一致。
- K/V tile 使用 ping-pong buffer；tile 粒度小于静态 packet chunk，不能把完整 64-token packet chunk 作为单个 core-local buffer。
- 每个 worker 处理本地 Q head 时，Q vector 必须先保留在 tile-local vector/register 中；扫描 KV row 时不能为每一行重复读取同一个 Q。
- DMA 读取下一块 K/V 时，当前块执行 score、mask、softmax running update 和 value accumulation。
- softmax 使用 running max、running sum、running output，不生成完整 attention score matrix。
- score reduction、running output correction、value accumulation 和 final normalize 都用 AIE vector 操作；head_dim 热循环不能退回 scalar 逐元素写法。
- GQA 下多个 query head 复用同一份 K/V tile，避免重复读取。
- RoPE 后的 Q/K 直接进入 attention stream，不写回 DDR。

目标是让 context 增大时 decode latency 按有效 KV tile 增长，而不是被格式转换或整块 cache 搬运放大。

### 6. Transformer block 内部 dataflow

fused dispatch 只是入口，block 内部必须按最终 dataflow 消除 DDR 往返：

- RMSNorm 输出直接进入 Q/K/V 或 FFN projection。
- Q/K/V projection、Q/K norm、RoPE、attention 必须在 tile/local stream 中衔接。
- attention output projection 直接接 residual add。
- FFN gate/up 输出直接接 SwiGLU，SwiGLU 输出直接进入 down projection。
- residual 和 norm buffer 作为 block 间唯一必要状态，其他中间 tensor 不作为长期 DDR 状态存在。

decode 的 Q/K norm + RoPE 正式实现使用 `RMSNormRoPE`：

- weighted RMSNorm worker 读取 activation 和 norm weight。
- norm output 通过 ObjectFifo 直接喂给 RoPE worker。
- RoPE worker 读取 angle row 后直接写入 Q/K stream。
- worker 拆分以满足 AIE tile input DMA channel 约束，不把 norm output 作为 DDR 边界。

decode 的 residual add + following RMSNorm 正式实现使用 `ResidualAddRMSNorm`：

- residual add worker 只读取 residual 和 update，产出更新后的 residual，并通过 ObjectFifo 把同一份 sum 喂给 norm worker。
- norm worker 读取 sum FIFO 和 norm weight，产出下一段 projection 需要的 norm output。
- attention residual 后直接产生 FFN 前的 norm output；FFN residual 后直接产生下一层 norm1 或 final norm output。
- worker 拆分以满足 AIE tile input DMA channel 约束，不把 residual sum 的 DDR 写回作为 norm 输入。

中间 buffer 不作为正式 dataflow 的边界；event trace 用来定位仍然存在的 DDR 往返，并推动它们进入同类 stream fusion。

## 性能观测要求

benchmark 和 event trace 是最终写法的验收约束，不是独立优化路线。正式实现必须能持续回答：

- decode 单 token 中 transformer、attention/KV、`lm_head`、cache update 各占多少。
- W4A16 GEMM 中 weight load、dequant、`aie::mmul`、output store 各占多少。
- DMA 与计算是否重叠，哪里有空泡。
- batch=1/4/8 下 column/core 是否持续有任务。
- context=128/512/1024 下 attention/KV 延迟是否按有效 tile 增长。

性能数据不能单独作为合入依据。任何 microkernel/dataflow 改写都必须同时通过端到端 token 输出验证；operator profile 只用于确认最终 dataflow 中的算子实现是否达标，不能产生独立代码路径，也不能证明 logits 正确。

观测入口固定为：

- `scripts/benchmark_qwen3_batch_decode.sh`
- `scripts/trace_qwen3_batch_decode.sh`
- `scripts/profile_qwen3_batch_decode_ops.py`

这些脚本必须始终测 `models.quantized_qwen3.qwen_npu` 和 packed W4A16 artifact，不能误跑未量化或旧模型路径。测量结果只用于判断最终 dataflow 是否达标。

性能验收只围绕最终执行形态：

- `lm_head` 必须在 decode fused ELF 内完成，不允许作为独立 dispatch 保留。
- W4A16 GEMM 必须使用 fused dequant `aie::mmul` microkernel，不允许 vector dot/reduce 作为正式路径。
- paired projection 必须复用 activation stream，不允许 K/V 或 gate/up 分别拉取同一份 activation。
- attention 必须从 packet KV tile 直接读取，不允许热路径 seq-major repeat 或 cache layout 转换。
- prefill 最后 chunk 必须在 ELF 内产生 logits，不允许额外 `lm_head` dispatch。
- event trace 必须证明主要 dataflow 中 DMA 与计算重叠；不能证明重叠的写法不作为最终方案。
- 端到端 token 输出正确性与 benchmark/profile 同时作为合入条件；单独的 operator profile 数字不能作为合入依据。

## 实现契约

正式代码只保留最终最高性能路径：

- 权重 artifact 只有 compressed W4 tile layout，不保留 bf16 GEMM 权重、不保留运行时转换格式。
- GEMM/LM head 只走 W4A16 fused dequant `aie::mmul` kernel，不拆成 dequant op 与 GEMM op。
- decode 只提交 transformer+`lm_head` fused ELF；Python 只负责准备输入、提交 ELF、读取 logits。
- prefill chunk 只提交 prefill ELF；最后 chunk 的 logits 在 ELF 内产生。
- KV cache 的正式 layout 必须是 attention 直接消费的 packet/tile layout；不允许 seq-major cache 和 attention cache 之间做热路径转换。
- batch decode 和单句 decode 使用同一套 fused path；batch 只是 M 维大小变化，不复制控制流。
- `quantized_qwen3` 和 `exported_llama3` 共享生成组织与 operator 主线，模型差异只来自 exported graph、config 和权重 layout。

## 代码组织约束

- 只保留一条正式实现路径。
- 不添加运行时兼容分支。
- 不用扩大 context window 解释性能问题。
- 不用 batch 增大替代 batch=1 优化。
- 不把 Python/Torch 中间操作留在 token decode 热路径。
- 不把 operator 试验代码留在正式模型目录。
- `quantized_qwen3` 和 `exported_llama3` 共享 torch2iron export 生成组织和 operator 主线，不复制优化。

## 成功标准

- decode 单 token 使用一个 transformer+`lm_head` ELF。
- prefill chunk 使用 ELF，最后 chunk 直接产生 logits。
- 所有线性层从 compressed W4 tile 读取权重。
- 反量化结果只存在于 tile-local vector/register 中。
- batch=1/4/8 都能有效使用多个 column/core。
- attention/KV 实现 DMA/compute overlap，不做整块 KV copy 或格式转换。
- event trace 能解释主要耗时，benchmark 能证明最终 dataflow 实际生效。
