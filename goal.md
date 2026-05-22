# torch2iron

PyTorch 模型自动编译到 AMD XDNA NPU 的工具链。

## 愿景：One ELF, One Model

整个模型（包括所有 transformer 层、attention、FFN、norm）编译为**一个 ELF 文件**，
运行时只需一次 dispatch 即可完成完整推理。没有多 kernel 调度开销，没有 host-device 往返。

核心使能技术是 **host-managed fixed KV cache**：KV cache 使用固定容量的状态
buffer，generate ELF 只读取固定形状的 past KV，并输出当前 token 的 present
K/V。host 在每次 dispatch 后把 present K/V 追加到 cache 的第 `t` 行，并更新
attention mask。ELF 本身保持完全静态——编译一次，永不 patch，永不 reload。

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
