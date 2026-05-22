# Optimized Attention Plan

## 背景

当前 `llama_3_2_1b` 的 decode attention 已经从 patched ELF/reload 迁移到静态
KV cache：generate ELF 固定读取 `(n_kv_groups, max_seq_len, head_dim)` 的 past
cache，当前 token 的 present K/V 在 ELF 内部走固定槽位，host 在 dispatch 后把
present K/V 追加到持久 cache。

这个方向是正确的，但当前 attention 实现仍然不是最终形态。现在的 decode attention
还保留了通用算子拼接的结构：

1. 当前 K/V 被写到固定槽位。
2. KV group cache 被 repeat 到 query head 视角。
3. 通用 GEMV 计算 QK score。
4. mask + softmax。
5. V 侧再做格式转换。
6. 通用 GEMV 计算 PV context。

这会产生几个问题：

- GQA 的 K/V 被物理 repeat，浪费带宽和 scratch。
- score、weights、repeated K/V、transposed V 等中间结果被显式物化。
- attention 输入格式反过来约束了 cache layout，导致需要额外格式转换。
- column 并发主要来自通用 GEMV，而不是 attention 本身的 head/KV-group 并发。

目标是把这条路径替换成 Llama decode 专用的 chunked KV attention op：直接消费
group-major KV cache，不 repeat K/V，不物化完整 score/weights，用 online softmax
按 chunk 产生 context。

## 参考实现

IRON 里已经有三类可参考的实现。

### 1. D1 fixed-cache attention

路径：

```text
~/projects/IRON/iron/applications/new-mega/experiments/d1_single_layer/design.py
~/projects/IRON/iron/applications/new-mega/experiments/d1_single_layer/d1_fixed_attention.cc
```

价值：

- 已实现 fixed max-cache + runtime mask。
- 已实现 chunked QK + online softmax + PV。
- `packed_chunk = [K_chunk, V_chunk, mask_chunk]`，kernel 内维护
  `running_max`、`running_sum` 和 `acc[head_dim]`。

限制：

- 只有一个 Worker，串行遍历所有 Q heads。
- 适合作为算法正确性模板，不适合作为最终 column 并发拓扑。

### 2. Qwen3 persistent attention2

路径：

```text
~/projects/IRON/iron/applications/qwen3_0_6b/persistent/attention_design.py
~/projects/IRON/aie_kernels/generic/qwen3_attention.cc
```

价值：

- `attention_columns=2` 按 KV head shard 分 column。
- 每个 column 处理自己的 KV heads 和对应 Q heads。
- 不做物理 K/V repeat，而是在 worker 内处理 GQA repeat。
- 两列路径里把 score + mask + softmax 合进同一个 Worker。
- K/V cache tap 按 `(layer, kv_head, seq, head_dim)` 连续块读取。

限制：

- 针对 Qwen3-0.6B：`q_heads=16`、`kv_heads=8`、`head_dim=128`、
  GQA repeat=2。
- 部分 position/mask 仍是编译期常量路径，runtime metadata 方案曾出现 timeout/NaN。
- 它是 Qwen3 persistent graph 的一部分，不是可直接复用的 Llama op。

### 3. new-mega phase-owned lanes

路径：

```text
~/projects/IRON/how-to-write-megakernel-in-IRON.md
~/projects/IRON/iron/applications/new-mega/production/phase_owned_stages.py
~/projects/IRON/iron/applications/new-mega/production/phase_owned_kernels.cc
```

价值：

- 使用固定 `num_lanes=8` 的 Worker 状态机，资源随 lane 数增长，而不是随
  `num_layers * phases` 增长。
- 8 lane 广播/聚合使用 `fabric_group_size=4`，即两组 L2 forward/join，避免
  8 路直接 fanout 耗尽 shim endpoint。
- 适合作为后续 full megakernel 的拓扑模板。

限制：

- 这是更大的 phase-owned decode body，不应直接搬进当前 Llama 单 op 优化。
- 对当前任务更有用的是 lane mapping、L2 broadcast/join、phase packet 这些模式。

## Llama Attention 目标设计

Llama 3.2 1B 的关键形状：

```text
n_heads      = 32
n_kv_groups  = 8
head_dim     = 64
gqa_repeat   = n_heads / n_kv_groups = 4
```

这非常适合 8-lane attention：

```text
lane 0 -> KV group 0 -> Q heads 0..3
lane 1 -> KV group 1 -> Q heads 4..7
...
lane 7 -> KV group 7 -> Q heads 28..31
```

每个 lane 只读取一个 KV group 的 cache chunk，并在 lane 内计算 4 个 Q heads 的
context。这样可以自然消除 K/V repeat。

### 输入输出布局

正式目标 cache layout：

```text
K cache: [n_kv_groups, max_seq_len, head_dim]
V cache: [n_kv_groups, max_seq_len, head_dim]
Q:       [n_heads, head_dim]
mask:    [max_seq_len] 或 [n_heads, max_seq_len]
out:     [n_heads, head_dim]
```

优先选择一维 mask：

```text
mask[seq] = 1 表示该 cache 行有效
mask[seq] = 0 表示 padding/future
```

如果后续需要 per-head mask，再扩展为 `[n_heads, max_seq_len]`。当前 causal decode
场景下所有 heads 的有效长度一致，一维 mask 足够。

### Kernel 计算

每个 lane 的逻辑：

```text
for local_q in 0..3:
    q_head = lane_id * 4 + local_q
    state.max = -inf
    state.sum = 0
    acc[head_dim] = 0

    for chunk in 0..num_chunks:
        load K_chunk[lane_id, chunk]
        load V_chunk[lane_id, chunk]
        load mask_chunk[chunk]

        scan chunk scores:
            score[row] = dot(Q[q_head], K[row]) * scale
            skip mask=0
            chunk_max = max(score)

        online softmax merge:
            new_max = max(state.max, chunk_max)
            rescale old acc/sum
            for valid row:
                weight = exp(score - new_max)
                acc += weight * V[row]
                chunk_sum += weight
            state.max = new_max
            state.sum = old_sum * correction + chunk_sum

    out[q_head] = acc / state.sum
```

后续可优化为一次 chunk scan 同时服务 4 个 Q heads，复用同一个 K/V chunk 的 load。
第一版先保持每个 local Q head 串行，降低正确性风险。

### 数据搬运

第一版可以让 host 预打包 lane-local chunk：

```text
lane_packet = [K_chunk, V_chunk, mask_chunk]
```

这和 D1 实验一致，便于快速验证 correctness。

正式集成时应直接用 cache 的 group-major layout 生成 TAP：

```text
base = kv_group * max_seq_len * head_dim + chunk_start * head_dim
shape = [chunk_size, head_dim]
stride = [head_dim, 1]
```

这样 host 不需要每 token 重新打包完整 attention 输入，只需要维护固定 KV cache 和
mask。

## 实验路线

每个实验使用独立目录，不通过一个大实现加开关混用。

当前进展：

- E1/E2/E3/E4/E5 已作为独立实验目录完成验证，正式代码接入后实验目录已清理。
- E1 证明 Llama `head_dim=64`、GQA repeat=4、runtime mask 的单 lane chunked
  online softmax 正确。
- E2/E3 证明 2-column 到 8-lane 的 KV-group 并发可行，E3 覆盖完整
  8 KV groups / 32 Q heads。
- E4 证明 attention worker 不能直接同时接 `q+k+v+mask` 四路输入：四输入 direct
  worker 在 1 lane 下被 AIECC 报 `number of input DMA channel exceeded`，8 lane
  join 版本也无法完成 placement。
- E5 是正式采用的形态：保持两输入资源形态，每个 KV group/chunk 的 packet 只传一次，
  在 worker 内一次更新 4 个 local Q heads。该实现已收敛为正式
  `LlamaChunkedAttention` operator，并在 NPU 上通过
  `max_seq_len=512, chunk_size=64` 边界有效长度检查，packed 输入从 E3 的
  2,113,536 个 bf16 降到 528,384 个 bf16。
- 正式 decode 已接入 E5：`src/models/llama_3_2_1b/llama_npu.py` 中旧的
  repeat/GEMV-score/mask-add/softmax/V-transpose/GEMV-context 子图已替换为
  `LlamaChunkedAttention`，并新增 per-layer `packet_cache`。正式 fused MLIR/ELF
  已编译到 `build_elf/seq512/fused_op.elf`。第一次端到端运行暴露出 full Llama
  fused scratch buffer 需要约 2.49GB host BO，超过当前约 2GB memlock/host BO 上限；
  已通过 LM head 分离把 decode logits 投影移到 CPU，正式短 prompt decode 现已跑通。

### E1: 单 lane 单 KV group attention

目的：

- 从 D1 fixed attention 抽出 Llama `head_dim=64` 版本。
- 输入一个 KV group、4 个 Q heads、若干 cache chunks。
- 验证不 repeat K/V 的 GQA attention 与 PyTorch reference 一致。

完成标准：

- `max_abs` 在 bf16 容忍范围内。
- position 边界覆盖：0、1、63、64、127、128、max_seq_len-1。
- mask 对 padding/future 生效。

### E2: 2-column attention prototype

目的：

- 复刻 Qwen3 `attention2` 的 column shard 模式。
- 每列处理 1 个或多个 KV groups。
- 验证 ObjectFifo split/fill/drain、cache TAP、context 拼接方式。

完成标准：

- 至少覆盖 2 个 KV groups、8 个 Q heads。
- 不出现额外 repeat buffer。
- NPU 输出和 PyTorch reference 匹配。

### E3: 8-lane Llama GQA attention

目的：

- 8 lanes 对应 8 KV groups。
- 每 lane 输出 4 个 Q heads 的 context。
- 使用固定 chunk size，例如 64 或 128。

完成标准：

- 32 heads 全量 context 匹配 reference。
- Preflight 不超过 tile input/output 和 ObjectFIFO 资源限制。
- 与当前正式 decode attention 做同 prompt、同 token 的延迟对比。

### E4: Cache TAP 直接读取

目的：

- 去掉 host 预打包 `lane_packet`。
- 直接从正式 cache layout 读取 K/V chunk。
- mask 作为独立 runtime buffer 或 lane-shared stream。

完成标准：

- host 每 token 只追加 present K/V slice 和更新 mask。
- attention op 不需要 seq-major/repeat 转换。
- 性能不回退。

当前结论：

- 直接把 `K cache`、`V cache`、`mask` 做成三个独立输入 FIFO 会让 attention
  worker 变成四输入：`q+k+v+mask`。AIE compute tile 输入 DMA 通道不够，1 lane
  都无法通过资源分配。
- 用 `ObjectFifo.join` 在图内把 K/V/mask 拼成 packet，也会让 8-lane graph 在
  endpoint/placement 上过载。
- 因此正式路径不能让 attention worker 直接消费三份独立 cache 输入；必须把 cache
  状态改成 chunk packet layout，或引入更大的 phase-owned/fabric-group
  megakernel 数据流。

### E5: One-packet-per-group chunked attention

目的：

- 保留 E3 已验证的 8-lane 并发形态。
- 每个 lane 每个 chunk 只读取一次 `[K_chunk, V_chunk, mask_chunk]`。
- lane 内一次更新 4 个 Q heads 的 online softmax state，去掉 E3 中按 local Q
  head 重复传 K/V/mask packet 的 4 倍输入带宽。

完成标准：

- 32 heads 全量 context 匹配 reference。
- NPU 通过 `max_seq_len=512, chunk_size=64` 的边界有效长度检查。
- packed 输入大小是 E3 的 1/4。

## 正式集成计划

实验通过后，将经验应用到正式代码，然后清理实验代码。

正式代码目标：

1. 新增 Llama decode attention 专用 IRON op。已完成：`LlamaChunkedAttention`。
2. 在 `src/models/llama_3_2_1b/llama_npu.py` 中替换当前 decode attention 子图。已完成：
   decode runlist 现在使用 `LlamaChunkedAttention`。
3. 保留现有 static KV cache binning：512/1024/2048 等静态 `max_seq_len` 变体。
4. 保持 host-managed present K/V writeback，不回到 ELF patch/reload。
5. 删除 repeat K/V、transpose V、显式 score/weight 中间 buffer。已完成于 decode
   runlist。
6. 优先采用 E5 的 packetized cache layout：

```text
packet_cache[layer][kv_group][chunk] =
  [K_chunk, V_chunk, mask_chunk]
```

这样 formal attention op 仍然只有两路输入：`Q group` 和 `packet chunk stream`。
host 或 fused decode 只需要更新当前 token 对应的 K row、V row 和 mask row，不需要
每步重打包完整 cache。

当前正式验证状态：

- `python -m compileall src` 通过。
- `git diff --check` 通过。
- E5 独立 NPU recheck 通过：`valid_length=1/64/512` 均 pass；随后实验目录已清理。
- packet layout offset 自检通过。
- 正式 `llama_npu.py --prompt-len 8 --num-tokens 2` 跑通：
  输出与 CPU baseline 一致（`SCENE I. The court`），TTFT 约 1.480s，
  decode 约 0.365s/token。
- 正式 `llama_npu.py --prompt-len 16 --num-tokens 3` 跑通：
  输出与 CPU baseline 一致（`SCENE I. King Leontes and`），TTFT 约 1.580s，
  decode 约 0.349s/token。
- 为避开当前环境的约 2GB host BO 上限，decode LM head 暂时在 CPU 上执行；这不是
  attention 算法限制，后续可以通过更细粒度的 weight staging 或 separate NPU LM
  head 恢复到 NPU。

正式验证：

```text
python -m compileall src
git diff --check
短 prompt 3/8/16 tokens 推理
同 prompt 对比 PyTorch 或当前 baseline token id
记录 TTFT、decode ms/token、NPU dispatch time
```

性能判断：

- 最低要求：结果正确，decode 不慢于当前 static KV baseline。
- 第一目标：去掉 repeat/format conversion 后，seq512 decode 明显下降。
- 第二目标：seq1024/seq2048 下 chunked online softmax 的带宽收益可测。
- 第三目标：8-lane 版本优于 1-lane/2-column prototype。

## 风险与处理

### 1. 8 lanes 不一定最快

IRON 的 B1 GEMV scaling 实验已经证明 column scaling 非单调。8 lanes 可能被
L2 broadcast/join、endpoint、BD 或 L1 pressure 抵消收益。

处理：

- 保留 1-lane、2-column、4-column、8-lane 独立实验结果。
- 用测量选择正式策略，不默认全开 8 lanes。

### 2. Mask 作为 runtime metadata 可能不稳定

Qwen3 曾尝试把 position/valid length 夹带到 attention stream，出现 timeout/NaN。

处理：

- 第一版使用普通 mask buffer，不做复杂 metadata 通道。
- 避免动态 TAP offset 和 runtime BD patch。
- position 只通过 mask 和 RoPE 输入影响 attention。

### 3. Kernel 一次处理 4 Q heads 可能寄存器压力过高

每 lane 同时维护 4 个 `acc[64]` 会增加 local memory/register pressure。

处理：

- 第一版 lane 内串行 4 个 Q heads。
- 通过 correctness 和 baseline 后，再尝试 2-head 或 4-head fused update。

### 4. Direct cache TAP 可能暴露 stride/BD 限制

之前 seq-major 实验曾遇到 stride 限制。

处理：

- 正式 cache layout 使用 group-major `[kv_group, seq, head_dim]`，让 chunk 内
  `[seq, head_dim]` 连续。
- 避免 0-stride repeat。
- 如果 direct TAP 受限，先接受 host 预打包作为实验路径，但正式目标仍是 direct TAP。

## 当前推荐下一步

清理实验代码，并继续优化 decode LM head/weight staging。

理由：

- E1/E2/E3 已证明 chunked online softmax 和 8-lane GQA correctness。
- E4 证明 formal code 不能把 K/V/mask 三份独立 cache 直接接到 attention worker。
- E5 在保持两输入 FIFO 资源形态的同时去掉 E3 的 per-local-Q packet 重复，已经接入
  正式 decode，并收敛为 `LlamaChunkedAttention`。
- 正式代码已经新增 per-layer `packet_cache` scratch buffer，并用 fixed-slot
  `StridedCopy`/host partial sync 维护当前 token 对应的 K/V/mask row。
- 下一步不是再改 attention 算法，而是解决 LM head/weight staging，避免 CPU logits
  成为新的 decode 瓶颈。
