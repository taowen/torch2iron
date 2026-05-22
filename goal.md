# torch2iron

PyTorch 模型自动编译到 AMD XDNA NPU 的工具链。

## 愿景：One ELF, One Model

整个模型（包括所有 transformer 层、attention、FFN、norm）编译为**一个 ELF 文件**，
运行时只需一次 dispatch 即可完成完整推理。没有多 kernel 调度开销，没有 host-device 往返。

核心使能技术是 **host-managed fixed KV cache**：KV cache 使用固定容量的状态
buffer，generate ELF 只读取固定形状的 past KV，并输出当前 token 的 present
K/V。host 在每次 dispatch 后把 present K/V 追加到 cache 的第 `t` 行，并更新
attention mask。ELF 本身保持完全静态——编译一次，永不 patch，永不 reload。

### Chunked KV Cache 计算模型

**编译时：**
- Past KV cache buffer 形状固定为 `(n_layers, 2, n_kv_groups, max_seq_len, head_dim)`，
  作为 host 维护的持久状态；generate ELF 看到的是固定 shape，不包含动态写入 offset
- 当前 token 的 Q/K/V 在 generate ELF 内部计算；当前 K/V 以固定 shape 的 present
  output 暴露给 host
- Attention mask buffer 形状固定为 `(n_heads, max_seq_len)`，作为输入状态表达哪些
  cache 行有效
- Attention GEMV 可以固定计算 `max_seq_len` 个 score；更高效的实现可编译多个
  generate 变体（例如 256/512/1024/2048）按实际上下文选择最小可用容量

**每 token 运行时（host 侧）：**
1. 准备当前 token 输入、RoPE/position 信息，以及固定容量 attention mask
2. 使用同一个 generate ELF dispatch；不 patch ELF，也不重建 `xrt::hw_context`
3. NPU 输出 logits 和当前 token 的 present K/V
4. host 将 present K/V 写入持久 past cache 的第 `t` 行，并把有效 token 计数加 1
5. 更新下一次 dispatch 使用的 attention mask：past cache 中已追加的历史行可见，
   padding 行被 mask 掉；本次当前 K/V 在 NPU 内走固定 present 分支

**NPU 侧计算（固定不变）：**
1. 计算当前 token 的 Q/K/V
2. 用当前 Q 对固定容量 past K 做 attention score；当前 K 作为固定的 present 分支
   参与同一次 attention，而不是写入动态 cache offset
3. Past 分数 + attention mask（无效位置变为 `-inf`）；当前 present 分支始终有效
4. Softmax：`-inf` 位置输出 0 权重，有效位置正常归一化
5. 用 softmax 权重对 past V 和当前 V 做加权求和，得到 attention context
6. 输出 logits 和当前 token 的 present K/V，供 host 追加到 cache

### 浪费分析

**计算浪费（GEMV over padding）：**
- 在位置 t，attention GEMV 计算 max_seq_len 个乘加，但只有 t+1 个有效
- 浪费率 = (max_seq_len - t - 1) / max_seq_len
- 例：max_seq_len=2048，t=64 时浪费 97%；t=1024 时浪费 50%；t=2000 时浪费 2%
- 注意：**原始 patched ELF 写法同样浪费**——GEMV 的 M 参数编译时就是 max_seq_len，
  softmax patching 只是告诉 softmax 截断到 context_len，GEMV 本身的计算量不变
- Generate 变体可以缓解这类浪费：为不同 cache 容量分别编译静态 ELF，运行时按
  `prompt_len + reserved_response_len` 选择最小可用容量

**带宽与同步：**
- 算法上不要求每 token 同步整个 KV cache。理想实现中，past cache 是持久 XRT BO，
  host 只写入并同步新增 present K/V 对应的 cache slice
- 以 llama_3.2_1b 为例，新增 K/V 行大小为：
  `16 layers × 2(K+V) × 8 kv_groups × 64 head_dim × 2B = 32KB/token`
- Attention mask 的完整大小是 `32 heads × 2048 × 2B = 128KB`；实现上也可以只更新
  边界或使用更紧凑的有效长度参数
- 如果 runtime 只能把 KV cache 放进 monolithic `input_buffer` 并每次调用
  `input_buffer.to("npu")`，才会退化为全量同步：
  `8 kv_groups × 2048 seq × 64 dim × 2B × 2(K+V) × 16 layers = 64MB/token`
- XDNA NPU 与 CPU 共享系统内存，`to("npu")` 主要是 cache coherency 操作而非 PCIe
  DMA；但是否全量同步仍然是 runtime 设计问题，不是 chunked KV 算法本身的要求

**对比 patched ELF 的开销：**
- Patched ELF：KV cache 在 scratch 中（NPU 持久），每 token 只上传 x + rope_angles (~4KB)
- 但每 token 需 `reload_elf()`：重建 `xrt::hw_context`，延迟约数毫秒
- Static KV：无 reload；代价是 host 管理 past/present 状态，以及同步新增 K/V slice
  和 mask 更新。即使退化到全量 64MB sync，在集成 NPU 上也可能比 reload 便宜

**结论：** 正确目标不是“把 KV cache 作为普通 input_args 每 token 全量上传”，而是
“固定容量 KV state + host 追加 present K/V + 静态 generate ELF”。在集成 NPU
（共享内存）架构下，这能消除 per-token ELF patch/reload；在 PCIe 连接的离散
加速器上，必须避免全量 KV 同步，否则带宽代价会不可接受。

## 实现路径

将 `torch.export` 导出的 FX Graph（aten ops）自动映射为 IRON 框架的算子组合，
通过 `FusedMLIROperator` 编译为单个 ELF，实现单次 dispatch 执行完整模型推理。

## 为什么可行

- IRON 已提供参数化的标准算子（GEMV、RMSNorm、SiLU、ElementwiseMul 等），无需生成 kernel 源码
- `FusedMLIROperator` 已支持将多个算子串联编译为单个 ELF，算子间通过 L3 buffer 名共享数据
- 每个算子可以独立编译测试，组合后行为不变
- Chunked KV cache 使 ELF 保持静态（无需 per-token patching/reload）

## 核心思路

1. `torch.export(model)` 得到 aten op 图
2. Registry 将每个 aten op 映射为 IRON op 实例（带具体 shape/tile 参数）
3. 图中的 view/reshape 等零开销 op 折叠为 buffer 别名
4. 识别重复层结构，复用同一 op 实例（不同层只是 weight buffer 名不同）
5. 生成 `FusedMLIROperator` 的 runlist + input_args/output_args 声明
6. Attention 中的 KV cache 使用固定容量 host-managed state；generate ELF 输出
   present K/V，host 追加到 past cache，避免动态 offset patching

## 两阶段策略（参考 torch2vk）

torch2vk 的 quantized_qwen3 和 optimized_qwen3 体现了两个演进阶段：

**阶段一：自动映射（对应 quantized_qwen3）**
- aten op 1:1 映射为 IRON 算子，图中每个 op 对应 runlist 中一个条目
- RoPE 被 torch.export 分解为 slice → neg → cat → mul → add，各自独立映射
- 正确性优先，快速跑通端到端

**阶段二：算子融合（对应 optimized_qwen3）**
- 识别常见融合模式，用单个高效 op 替代多个原子 op
- 例如：RMSNorm + RoPE + Transpose → 单个融合 kernel
- 例如：Q/K/V 三个 GEMV → 单个 fused QKV projection
- 例如：Linear + Residual Add → 单个 matvec_add
- 例如：Gate + Up + SiLU + Mul → 单个 SwiGLU
- 减少 runlist 长度 = 减少 L3 round-trip 次数

## 相对 torch2vk 的架构优势

**无需 record/replay 双模式。** torch2vk 有一个根本性缺陷：eager record 阶段执行到 op N 时不知道后续计算图，
无法判断哪些 tensor 已死，无法提前释放显存。只有录完后做 liveness analysis（replay 模式）才能优化内存。
这导致必须维护两套执行路径，且首次执行的内存占用远高于稳态。

torch2iron 不存在这个问题——FusedMLIROperator 在编译时就持有完整 runlist，
`_calculate_buffer_layout()` 静态计算所有 buffer 的生命周期和偏移，
中间结果全部放在预分配的 scratch buffer 中。没有运行时动态分配/释放，没有 record/replay 分裂。

## 与 torch2vk 的类比

| torch2vk | torch2iron |
|---|---|
| Vulkan shader (GLSL) | IRON MLIROperator 实例 |
| ShaderRegistry | IRONOpRegistry: aten → IRON op |
| ReplayPlan（多 dispatch 打包提交） | FusedMLIROperator（多 op 编译为单 ELF） |
| 手写 fused shader（optimized） | IRON 融合 op 或自定义 kernel |
| record + replay 双模式 | 仅编译时静态规划，无运行时模式切换 |

区别：torch2vk 需要为每种 shape 生成 GLSL 源码，torch2iron 只需实例化已有参数化算子。
