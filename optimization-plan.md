# Optimization Plan

## 目标

当前优化主线调整为学习 FastFlowLM 的 decode 架构，并在 torch2iron 中实现同类高性能路径：

- 每个 transformer layer 是一个 fused layer engine，而不是跨所有 layers 的巨型 ELF。
- decode 热路径直接消费 packed Q4 权重，不先展开成 BF16 权重 BO。
- host 侧用 runlist/批量提交降低每 token 多层 dispatch 开销。
- layer 内部复用 hidden stream，连续完成 Q/K/V、attention、O、FFN gate/up、SwiGLU、down、residual/norm。
- KV cache 使用 attention 直接消费的分块 layout，只按有效 context tile 扫描。
- `lm_head` 可以暂时作为单独 packed-Q4 run；优先把 transformer layer 做到高利用率。

旧的“单 token 一个跨所有 layers 的 ELF”方案保存在 `legacy-old-plan.md`。它只有在跨层 hidden 能真正保留在 AIE/NoC 内、不经过 BO/DDR 边界时才有意义；当前更务实也更接近 FastFlowLM 的路线是一层一个高质量 fused layer kernel。

## 参考文档

FastFlowLM fused layer engine 的反编译依据保存在 `/var/home/taowen/projects/MyLM/tools/re/fused-layer-engine/README.md`。后续实现以这套文档为主要参考：

- `01-host-decode-flow.md`: host decode、runlist、layer/lm_head 边界。
- `02-transaction-patch-flow.md`: layer transaction 中 620 个 DDR patch/BD patch 的流向，尤其是 608 个 packed Q4 weight patch。
- `03-aie-cdo-topology.md`: PDI/CDO、row0/row1/row2..5 拓扑、program memory 聚类。
- `04-q4-weight-format.md`: Q4NX group32 chunk 和 projection patch layout。
- `05-aie2p-main-kernel-pseudocode.md`: 主 compute tile 的 online Q4 unpack/dequant/vector MAC 热路径伪代码。
- `06-kv-cache-and-attention-flow.md`: KV cache 和 attention dataflow。
- `07-open-questions.md`: 需要 AIE trace、完整 ISA disassembler 或自编 kernel 对照继续确认的问题。
- `08-bd-and-phase-trace.md`: layer transaction、CDO 静态 BD、lock/use_next chain 和 phase 推进。
- `09-current-answers.md`: 当前已确认的 tile 分工、Q4 MVM 分片、hidden replay、KV layout 和 RTP 语义。
- `10-phase-level-contract.md`: projection phase 的 16-tile fabric、weight patch range、KV current write/history read 公式和 start/config 写入契约。
- `11-complete-dataflow-contract.md`: 按 projection、hidden replay、Q4NX MVM、weight DMA、attention、KV、RTP 和中间驻留逐项标注 hard/strong/open。
- `12-kv-scan-bd-contract.md`: KV scan 的四平面 DDR descriptor、row1 memtile lock ring、16-token rounded history read 和 tail-mask 契约。

可复用诊断脚本：

- `tools/re/layer_contract_summary.py`: 从 layer transaction trace 汇总 projection/KV patch contract。
- `tools/re/aie_cdo_reg_writes.py`: 抽取 CDO core/memtile stream-switch register writes，后续用于解码 ObjectFIFO/stream route。
- `tools/re/kv_bd_contract.py`: 从 layer transaction trace 汇总 KV current write 和 history scan 的 BD contract。

算法到硬件的设计原则保存在 `/var/home/taowen/projects/MyLM/the-algo-maps-to-hardware.md`。后续 fast_qwen3 的 operator 和 fused layer 设计也必须对齐这份文档：

- XDNA2 是 8 列 x 6 行 tile 网格；row0 是 shim/IO，row1 是 memtile，row2-5 是 compute tile。
- compute tile 本地存储只有 64KB，程序和数据共用；decode 中间向量适合留在 tile/local stream，完整 BF16 权重不适合物化。
- decode 是权重带宽受限场景，优先策略是 packed Q4 在线反量化、output-dimension partitioning、避免 K 维切分后的跨 tile reduce。
- 权重和 KV 的 DDR DMA 必须按列并行并尽量 double-buffer，让 DMA 和 vector MAC 重叠。
- runtime 应保持静态拓扑，只动态 patch DDR 地址和少量 RTP；不要为了动态性在 host 侧重建复杂流程。
- 算子融合边界按数据驻留决定：能放进 tile/local stream 的 decode 中间向量应融合，超出本地容量或会破坏 DMA/compute overlap 的路径才拆分。

## 已确认的参考事实

基于 FastFlowLM 公开 runtime、符号和控制序列分析，可以确认以下设计点：

- decode 主路径不走 standalone `mm.xclbin`、`dequant.xclbin`、`attn.xclbin`。
- decode 每层执行 `layer.xclbin`，packed Q4 权重 BO 直接作为 layer input。
- `layer.xclbin` 是真实 fused layer engine：AIE PDI/CDO 直接配置 compute tile program memory、memtile DMA BD 和 shim/control，不是多个 standalone xclbin 的 host transaction 包装。
- 参考 topology 是 8 列 x 4 行 compute core。8B layer 中 row 2..5 是 compute rows，row 1 配 memtile DMA，row 0 是 shim/control；主计算 kernel 在中间 compute tiles 上高度复用，左右边界和汇聚/control kernel 较小。
- `layer.xclbin` 不是所有 projection 同时并发，而是 phase 内空间并行、projection 间时间复用。Q/K/V/O/up/gate/down 都复用中心 `c2..c5/r2..r5` 的 16 个主 Q4 online MVM tile。
- projection MVM 的 host patch 按 output rows 分片，不按 K 分片。一个 512-wide output block 是 `4 columns * 2 patches/column * 64 rows`；强推断每列 4 个 compute rows 各吃一个 32-row Q4NX chunk，K 在 core 内串行消费，没有 projection 跨 tile K-reduce 证据。
- 一个 512-wide output block 的列分片可实现为 `c2=0..127, c3=128..255, c4=256..383, c5=384..511`。每列两个 64-row patch，再拆成四个 32-row Q4NX chunk 交给 `r2..r5`；row 内低高顺序还没有硬证据，当前实现先按低到高连续映射。
- 8B transformer block 的 weight patch range 是 `Q 0..63, K 64..79, V 80..95, O 96..159, up 160..351, gate 352..543, down 544..607`。
- Q/K/V 是三个顺序 projection phase，共享同一份 `hidden_norm` replay；up/gate 也是两个顺序 projection phase，共享同一份 `ffn_norm` replay。当前 torch2iron 的 QKV/up-gate interleaved operator 只是验证 Q4NX online-MVM 和 hidden fanout 的临时边界，不是最终 layer phase schedule。
- hidden/normed hidden host 只 send 一次，row1 memtile BD chain 表明 hidden 在 memtile 内 replay/broadcast 到 `c2..c5`，Q/K/V 不各自重读 DDR。
- layer phase 同步主要靠 static BD `use_next` chain、lock acquire/release、core `acq` 指令和少量 dynamic writes，不是 host 逐个启动小 kernel。
- weight shim BD 使用 even/odd 双 bank：`bd1/bd2` 和 `bd9/bd10` 交替，说明有 DMA/compute overlap 的静态 pipeline。
- KV cache 的四个物理 plane base 是 `0/0x400000/0x800000/0xc00000`，token stride 是 `0x400` bytes，plane size 是 `0x400000`。
- KV plane 语义绑定是 `0x000000 -> k03`、`0x800000 -> k47`、`0x400000 -> v03`、`0xc00000 -> v47`。history read 按物理地址顺序读 `0x0,0x400000,0x800000,0xc00000`，current write 顺序才暴露语义 K/V 和 03/47 绑定。
- KV current write 公式是 `plane_base + (L - 1) * 0x400`，BD word0 是 `0x100` dwords；history read 是 `ceil(L / 16) * 0x1000` dwords，即每 plane `ceil(L / 16) * 0x4000` bytes。BD length 是 dword count 而不是 byte count，非 16 对齐 tail 必须由 attention L mask 处理。
- KV scan 不是把 K/V pair 合成一条 DDR BD chain。`arg4` 是四个 plane 的独立 descriptor：`k03/v03/k47/v47`；dynamic shim BD 的 `use_next=0`，真正的循环和 double buffering 在 row1 memtile 的静态 BD/lock ring 上。
- row1 memtile plane reader 至少有两套 lock ring：ring A 使用 `64 -> 65 -> 66 -> 64`，ring B 使用 `67 -> 68 -> 69 -> 67`。其中 4096 dwords 正好是 `16 tokens * 4 heads * 128 dim * bf16`，2048 dwords 是对应半 tile/分流段。
- 二进制里的 ObjectFIFO 语义落到显式 BD、`use_next`、lock acquire/release 和 worker `acq/release`，复刻时不能假设有隐式无限 FIFO。
- attention 的 DDR-visible KV tile 是 `[16 tokens][4 KV heads][128 dim] bf16`；worker 前 row1 memtile 还会做 reshape/split，GQA head 到具体 worker 的 exact mapping 仍需数值探针或 stream switch decode 继续确认。
- Q4 权重没有在 load model 时全量 dequant 成 BF16 BO。
- 如果 layer 内部存在 BF16 展开，也只应是 AIE local tile/register/stream 级别的瞬时数据。
- AIE core 程序里已经确认 online Q4 hot path：`vlda` 读 packed Q4、`vunpack/vups` 拆 nibble、`vsub.f` 做 zero/offset、`vconv.bf16.fp32`、`vmul.f/vmac.f` 做 vector MAC。当前优化应优先复刻显式 vector unpack + vector MAC pipeline，不再把 `aie::mmul` 作为 decode 单 token 的必要路径。
- Q4NX chunk 固定为 `32 output rows x 256 input cols`，每 chunk 5120 bytes。
- chunk 内部布局是：256 个 bf16 scale、256 个 bf16 zero-point、4096 bytes packed int4。
- dequant 公式是 `w = (q4 - zero_point) * scale`。
- group size 是 32，也就是每个 output row、每 32 个 K 共享一组 scale/zero-point。
- Qwen3 layer weight patch 顺序是 `Q -> K -> V -> O -> up -> gate -> down`。
- transformer block patch 数按 projection 顺序成立：8B 是 `64 + 16 + 16 + 64 + 192 + 192 + 64 = 608`，0.6B 是同公式下的 176。
- Q/K/V 是连续 weight stream 区间，但在 layer 内是三个顺序 phase，不是 host 侧三个独立 dequant/GEMM kernel。
- up/gate 是连续 FFN projection 区间，但在 layer 内是两个顺序 phase，共享同一份 FFN norm activation replay。
- KV cache 分为 `k03/v03/k47/v47` 四个物理 base 顺序，对应 8 个 KV heads 的 0..3 和 4..7 分组；语义顺序不能和物理地址顺序混用。
- attention 通过当前 L/RTP 按有效长度读 cache，不应扫描完整 max context。
- runlist 主要减少 host submit/syscall 开销；layer 之间仍有 hidden state 数据依赖，不能真正并行计算。
- prefill 可先维持 chunk/big-M GEMM 路径；短期最值得复刻的是 decode layer fused online-Q4。

## 新的最高优先级

### 1. Q4NX packed layout 支持

正式权重格式先向 Q4NX group32 靠拢，而不是继续以 generic W4A16 group128 GEMM 为中心：

- 磁盘 artifact 保存 layer kernel 直接消费的 packed Q4 patch。
- chunk 粒度固定为 `32 out x 256 in`。
- scale、zero-point、int4 数据保存在同一个 patch stream 中。
- host 热路径不做 transpose、repeat、dequant 或 BF16 权重物化。
- Q/K/V、O、up、gate、down 按 projection-major 和 output shard 顺序连续保存。

第一阶段可以写 converter，把现有 safetensors/quantized artifact 转成 Q4NX-like patch layout，并用 CPU dequant reference 对拍。

### 2. Q4NX online-MVM projection 原型

先实现一个最小可验证 operator：

- 输入：pre-normalized hidden，shape 先固定为 batch=1 decode。
- 权重：Q/K/V 三段 Q4NX group32 patch。
- 输出：Q、K、V 三个 tensor。
- dataflow：hidden stream 进入一次，验证同一份 activation 能被多个 Q4NX online-MVM worker 复用。
- dequant：在 AIE tile 内 unpack int4，应用 scale/zero-point，立即累加。
- 不生成完整 BF16 weight tensor。
- 不接 attention、不接完整 layer、不接 runlist，先验证 layout、数值和单 op 性能。

这个原型是后续所有 layer fusion 的基础。它回答两个关键问题：Q4NX patch layout 是否理解正确，以及 hidden activation 是否能在 IRON dataflow 中被 Q/K/V 复用。FastFlowLM 的最终 layer schedule 是 Q、K、V 三个顺序 phase 共享 `hidden_norm` replay；当前 interleaved QKV operator 不能直接等同于最终 phase schedule。

### 3. Fused layer engine

QKV 原型通过后，把目标从 generic GEMM 转为 layer-specific engine：

- layer 输入：hidden state、rope/position、packed layer weight、KV cache、norm weights。
- layer 输出：下一层 hidden state、更新后的 KV cache。
- layer 内部顺序：
  - RMSNorm -> Q、K、V 三个顺序 online-Q4 projection phase，共享 `hidden_norm` replay。
  - Q/K norm + RoPE。
  - streaming attention，按有效 KV tile 扫描。
  - O online-Q4 projection。
  - residual add + FFN RMSNorm。
  - up、gate 两个顺序 online-Q4 projection phase，共享 `ffn_norm` replay。
  - SwiGLU。
  - down online-Q4 projection。
  - residual add。
- 中间 Q/K/V、attention output、up/gate、SwiGLU output 不作为 host-visible DDR 边界。

这条路线优先优化 batch=1 decode。batch 只是 M 维扩展，不能用 padding batch 掩盖单请求利用率问题。

### 4. Runlist runtime

layer engine 成熟后，再改 runtime：

- 每层一个 fused layer ELF/app。
- 每 token 使用 runlist 提交所有 layer runs。
- layer 间 hidden state 用 ping-pong BO 或固定 layer buffer 传递。
- 每层使用自己的 packed layer weight 和 KV cache view。
- `lm_head` 初期保留为 runlist 之后的单独 Q4 packed run。

如果 runlist 不直接可用，允许先用 Python/C++ for-loop submit 验证 layer kernel 本身收益。runlist 是 dispatch 优化，不应该阻塞 microkernel/dataflow 验证。

### 5. KV/attention streaming

attention 需要围绕 cache layout 重做，而不是依赖格式转换：

- KV cache 物理 layout 与 attention 读取顺序一致。
- KV heads 可按 `0..3` 和 `4..7` 分平面，匹配 GQA 复用。
- attention 只扫描有效 L，L 按 tile 对齐，例如 16-token tile。
- Q vector 在 tile-local register/vector 中复用，扫描 KV row 时不重复读取 Q。
- GQA 下多个 Q heads 共享同一份 K/V stream。
- softmax 使用 running max、running sum、running output，不生成完整 score matrix。
- event trace 必须能证明 KV DMA 与 score/value compute 有 overlap。

### 6. LM head

`lm_head` 不再是当前阶段最高优先级：

- 先接受它作为独立 packed-Q4 run。
- 确认它也直接消费 Q4 packed weight，不物化 BF16 weight。
- 后续再考虑放进 runlist或并入最终 token path。
- 是否并入 transformer layer 之后再决定，不能因为追求“一次 dispatch”牺牲 layer kernel 主线。

### 7. Prefill

prefill 暂时不作为第一优先级：

- 保留 chunked prefill 和大 M GEMM 路线。
- 确保最后 chunk 能产生 logits，避免不必要的额外 host work。
- 后续再考虑把 prefill 的 Q4 dequant 和 GEMM 更深融合。

decode 是当前性能差距最明显、也最能体现 packed Q4 online consumption 的路径，先集中资源复刻 decode layer engine。

## 实施顺序

1. 写 Q4NX group32 patch parser/converter 和 CPU reference dequant。已完成：`models.fast_qwen3` 会生成 `fast_qwen3_q4nx` artifact。
2. 做单层单 projection CPU 对拍，验证 `Q4NX::_q4nx_reorder` 等价布局。已完成：`scripts/run_fast_qwen3_qkv_smoke.sh` 会验证完整 Q/K/V reference 和第一 patch reference。
3. 实现 `q4nx_fused_qkv_projection` AIE operator，输入 raw hidden，内部做 RMSNorm，输出 Q/K/V。已完成：默认按 Qwen3-0.6B 的 `hidden_size=1024` 分 4 个 256-K chunk 流式累加 Q/K/V output patches。这个 operator 作为 Q4NX online-MVM 和 hidden fanout 验证边界保留；最终 layer schedule 应改为 Q、K、V 顺序 phase。
4. 对拍 Q/K/V 数值和 profile 单 op。已完成：`scripts/run_fast_qwen3_qkv_operator_smoke.sh` 会编译 full ELF、上 NPU 运行、与 CPU patch reference 对拍、输出 wall-time profile，并在 `--trace-size` 开启时解析 event trace。
5. 把 RMSNorm 合入 QKV projection 输入侧，减少 hidden DDR 边界。已完成：QKV operator 内部用 norm worker 计算 normed hidden，再 forward 给 output-patch workers。
6. 把 QKV projection 从单 core 单 output patch 扩成多 output patch 并行，开始使用多列 compute core。已完成：默认 8 个 output patches 映射到 8 个 compute columns。
7. 消除多 patch 版本中重复 hidden/RMSNorm DMA。已完成：hidden 和 RMSNorm weight 各传一次，QKV packed stream 不再重复 norm slice。
8. 实现 fused up/gate projection，复用同一份 FFN norm stream。已完成：`Q4NXFusedUpGateProjection` 使用同 QKV 一样的 norm worker + 8 output-patch workers，packed stream 顺序为 `up,gate`。这个 paired operator 是验证边界；最终 layer schedule 应改为 up、gate 顺序 phase，共享 `ffn_norm` replay。
9. 实现 layer-local attention/KV update，避免 Q/K/V 中间 DDR。进行中：
   `QwenChunkedAttentionCurrent` 已验证 8 个 KV group 的 current-aware
   attention，Q 和当前 K/V 合并成一路 `q_current` 输入，packet cache 作为另一路
   输入，避免 AIE tile 超过 2 路 input DMA。`QwenCurrentKVCacheWrite` 已作为
   独立小写入 operator 持久化同一个 current K/V slot；attention-current 和
   cache-writer 可以在同一个 fused ELF 中顺序执行。直接从 attention worker 输出
   完整 updated chunk 会因 tile-local SRAM 超限失败，因此不采用。`QwenQKVToQCurrent`
   已把 QKV projection patch 输出组装为 attention/cache-writer 共同消费的
   `q_current`，当前 8-patch projection 可覆盖前 2 个 KV groups。`Q4NXFusedQCurrentProjection`
   已把 group-major `Q4 + K2 + V2` packed stream 直接投影成 1 到 8 个 KV groups
   的 `q_current`，RMSNorm/hidden 只搬运一次，并已验证 8-group
   `projection -> attention` 稳定重入；`projection -> writer -> attention`
   的三段临时 runlist 当前第二次调用会 timeout。已尝试把 current K/V write
   合入 `QwenChunkedAttentionCurrent`：让 attention worker 自己额外输出
   `current K + current V + mask` 会在 1-group 下打坏 context 数值；拆成 row3
   writer worker 并从同一份 `q_current` FIFO 广播消费，1-group 可跑，但 8-group
   会让 `SequentialPlacer` 报 `Failed to find a tile matching column 0`。因此
   8-group 正式路径仍保留独立小 writer；真正 fused layer 需要手工 placement 或
   更底层的 layer engine 调度来表达这个 update stream，而不是继续在当前自动
   placer 上堆临时 FIFO。
10. 实现 attention output 后的 `o_proj` online-Q4 projection block。已完成：
    `Q4NXFusedLinearProjection` 已按 FastFlowLM phase-level contract 改为 output-row split：
    每个 64-row output patch 拆成两个 32-row Q4NX chunk workers，每个 worker
    一次 DMA 自己的 32-row/full-K packed Q4 buffer，并在 core 内串行消费 K。
    这替代了早期的 K-split/reducer 临时实现，避免了跨 tile K partial reduce。
    当前稳定覆盖 Qwen3-0.6B `o_proj` 的一个 8-patch、512-wide block，
    placement 对齐完整 `c2..c5/r2..r5` 主 projection fabric；完整 `o_proj`
    共有两个这样的 block。关键实现约束是 activation 从 c0 单独 ingress 并
    broadcast 到 16 个 projection workers，c2..c5 shim 只承载每列两个 weight
    patch ingress；weight 在 row1 按 64-row patch split 到两个 32-row workers，
    output 也按 64-row patch join 后 drain。
    `QwenChunkedAttentionCurrent ->
    Q4NXFusedLinearProjection` 两段边界已跑通。把
    `Q4NXFusedQCurrentProjection -> QwenChunkedAttentionCurrent ->
    Q4NXFusedLinearProjection` 三段直接串进当前自动 fused runlist 会 timeout
    或让 `o_proj` 输出错误，因此下一步必须做单层专用 dataflow/placement，
    不能继续堆临时 operator。
11. 组装单层 fused layer engine。进行中：
    下一步不是继续扩展自动 `FusedMLIROperator` runlist，而是写手工 phase dataflow：
    projection fabric 已有可运行 512-wide block，下一步要把它从 isolated
    `o_proj` operator 迁入 layer-local phase dataflow，并保留 row-order
    permutation probe；`Q4NXFusedLinearResidualProjection` 已把 512-wide
    `o_proj` block 和 residual add 合进同一个 ELF，验证 projection output
    可以进入后续 residual math 而不回到 Python；同一 fused call 已能串接
    两个 512-wide blocks，覆盖完整 Qwen3-0.6B hidden-size O residual 输出；
    `QwenCurrentKVPlaneWrite` 已开始按 FastFlowLM 四平面 contract 写 current
    K/V，物理平面顺序是 `k03, v03, k47, v47`，token stride 是
    `4 * head_dim`；`QwenPlaneAttentionCurrent` 已开始从四平面 cache 直接读
    K/V tile，current slot 仍从 `q_current` 旁路，`attend_seq_len` 作为有效行数处理 tail mask。
    当前高层实现已改成两个 plane-pair reader：`k03/v03` 服务 groups 0..3，
    `k47/v47` 服务 groups 4..7。runtime fill 用 group-major tap，row1/memtile
    split 只处理连续 group slice，不再把同一份 KV history 重复从 DDR 读 4 次。
    Q 输入和 context 输出也收敛到 plane-pair 粒度，避免 8 路 runtime fill/drain
    继续消耗动态 BD。实测 64-token/16-token tile 和 128-token/32-token tile 都能
    用 high-level IRON 编译运行并通过数值对拍。128-token/16-token tile 仍会因为
    20 个动态 DMA task 超过 BD 预算而失败；单级 8 路 split/join 又会撞 memtile
    DMA channel 限制，二级 ObjectFIFO link 也不合法。因此最终长 context reader
    仍应做 FastFlowLM 式 row1 memtile static BD chain + runtime patch descriptor，
    但 high-level IRON 已足够表达 coarse DDR read + memtile group fanout 的主形状。
    `Q4NXFusedQCurrentProjection -> QwenPlaneAttentionCurrent` 已在同一个
    fused ELF 中跑通，验证 packed-Q4 q_current projection 可以直接接四平面
    attention，旧 packet-cache attention 边界已有可替换的稳定 smoke。
    `QwenCurrentKVPlaneWrite -> QwenPlaneAttentionCurrent` 也已用 high-level
    IRON 在同一个 fused ELF 中跑通：`kv_plane` 作为 persistent external BO，
    writer 只更新 slot 0/5 的 current K/V row，整块 plane 回读与 CPU in-place
    reference 完全一致，随后下一 token 的 plane attention 能把这行作为 history
    读取。这个结果说明 current-row persistent write 不必过早下沉到低层 CDO；
    `Q4NXFusedQCurrentProjection -> QwenCurrentKVPlaneWrite ->
    Q4NXFusedQCurrentProjection -> QwenPlaneAttentionCurrent` 也已在 high-level
    IRON 中跑通，模拟两个 decode step：第一步 packed-Q4 projection 产生
    current K/V 并持久化到 plane cache，第二步 packed-Q4 projection 产生下一
    token 的 q_current，attention 读取刚写入的 history。未写入的 plane rows
    bit-exact 保持，写入 row 的差异只来自 projection 数值误差。下一步应把这个
    边界和 O projection/residual 边界收敛成单层 layer-local phase schedule，
    同时继续用 stream-switch decode 和数值探针收敛 attention edge fabric。
12. 用 Python for-loop 跑完整模型，确认端到端 token 输出正确。
13. 引入 runlist，把每 token 的多层 submit 合并。
14. 再优化 `lm_head` 和 prefill。

## 当前测量

`q4nx_fused_qkv_projection` 当前不是完整 Q/K/V projection，而是完整 K 宽度的一组 output patches：

- 输入 hidden：1024 bf16，未预先 RMSNorm。
- 输入 RMSNorm weight：1024 bf16，单独传入，不随 output patch 重复。
- 输入 Q/K/V packed Q4NX stream：每个 output patch 122880 bytes，按 patch-major、K chunk 交错为 `Q0,K0,V0,Q1,K1,V1...`。
- 默认输出：8 个 output patches，每个 patch 中 Q/K/V 各 64 bf16，shape 是 `[8, 3, 64]`。
- 数值：8-patch norm-worker 版本相对 CPU reference 的 `max_abs_error=0.0078125`。
- 性能：同一套代码下 1-patch full-ELF smoke 的 5 次 wall-time median 约 0.41 ms；4-patch median 约 0.89 到 1.05 ms；8-patch 最好一次 5 次 wall-time median 约 0.71 ms。norm-worker 版本 20 次样本 median 约 0.71 ms，最好一次约 0.66 ms，恢复确认 run 的 20 次 median 约 0.707 ms，短 run 仍有较大 wall-time 抖动。多列并行已经提高每 patch 吞吐，8 列比 4 列更接近目标。
- 失败实验：把每个 QKV worker 的 weight FIFO 从 30720B 拆成 3 个 10240B projection patch 会耗尽 BD，AIE lowering 报 `Allocator exhausted available buffer descriptor IDs`。下一步要在不增加 DMA task 数的前提下降低 tile-local memory 压力。
- 失败实验：runtime 仍一次搬 30720B 到 row1 memtile，再 split 成 Q/K/V 三个 10240B core FIFO，会让 QKV worker 同时拥有 `normed_hidden + Q + K + V` 四路输入，AIE resource allocation 报 `number of input DMA channel exceeded`。
- 失败实验：把默认并行度扩到 16 个 output patches，映射到 8 列 x 2 行 compute，并用两个 norm row-groups 分别 fanout 到 row2/row3，目前会在 `SequentialPlacer` 报 `Failed to find a tile matching column 0`。当前 IRON ObjectFifo fanout 写法还不能直接表达同列多 compute row 的 normed-hidden broadcast。
- 历史基线：未融合 RMSNorm 的 vectorized QKV patch 约 0.40 ms，scalar dot/reduce 约 70 ms。
- trace：8-patch norm-worker 版本开启 trace 后 routing 会报 `aie.masterset` destination 冲突；1-patch trace 能编译运行，但 Chrome trace 转换仍会返回 `UnboundLocalError: cycles`。当前不能把 trace run 的 wall-time 当性能指标。

`q4nx_fused_up_gate_projection` 当前是 FFN gate/up 前半段的 paired projection 原型：

- 输入 hidden：1024 bf16，未预先 RMSNorm。
- 输入 FFN RMSNorm weight：1024 bf16，单独传入，不随 output patch 重复。
- 输入 up/gate packed Q4NX stream：每个 output patch 81920 bytes，按 patch-major、K chunk 交错为 `up0,gate0,up1,gate1...`。
- 默认输出：8 个 output patches，每个 patch 中 up/gate 各 64 bf16，shape 是 `[8, 2, 64]`。
- 数值：8-patch norm-worker 版本相对 CPU reference 的 `max_abs_error=0.015625`。
- 性能：20 次样本 median 约 0.55 ms，min 约 0.50 ms。它比 QKV patch 轻，因为每个 K chunk 只含两个 projection，core-local weight buffer 是 20480B。

`q4nx_fused_q_current_projection` 当前是 decode attention 所需的 group-local projection 原型：

- 输入 hidden：1024 bf16，未预先 RMSNorm。
- 输入 RMSNorm weight：1024 bf16，单独传入。
- 输入 group-major Q/K/V packed Q4NX stream：每组按 `Q patch x4, K patch x2, V patch x2` 选择 projection patch，每个 patch 内按 K chunk 顺序搬运，每个 K chunk 是两个 Q4NX chunk 共 10240 bytes。
- 输出：每个 KV group 一个 `q_current`，shape 是 `[num_kv_groups, 512]`，包含两个 Q heads、current K、current V。
- 数值：单组单 op smoke 相对 CPU reference 的 `max_abs_error=0.005859375`；8-group 单 op smoke 的 `max_abs_error=0.01171875`。
- 性能：单组 3 次样本 median 约 0.44 到 0.47 ms；8-group 单 op smoke 的 3 次样本 median 约 1.20 ms。旧 packet-cache direct 8-group `projection -> attention` smoke 的 `attention_max_abs_error=0.0078125`，3 次样本 median 约 2.04 ms。新四平面 direct 8-group `projection -> plane attention` smoke 在 64-token/16-token tile 下 `attention_max_abs_error=0.015380859375`，3 次样本 median 约 1.89 ms；128-token/32-token tile 下 `attention_max_abs_error=0.007198333740234375`，单次样本约 2.34 ms；7-token tail 下 `attention_max_abs_error=0.029296875`。
- 已确认的限制：`projection -> attention` 可以重复执行；`projection -> writer` 也可以重复执行；但 `projection -> writer -> attention` 或 `projection -> attention -> writer` 的三段临时 runlist 第二次调用会 timeout。当前不把这个三段边界作为正式方向，后续 layer engine 应直接在 layer-local dataflow 中持久化 current K/V。

`q4nx_fused_linear_projection` 当前是 attention 后 `o_proj` 的 online-Q4 projection block 原型：

- 输入 activation：2048 bf16，对应 Qwen3-0.6B attention context。
- 输入 `o_proj` packed Q4NX stream：每个 output patch 81920 bytes，按 patch-major、K chunk 顺序保存。
- 输出：默认 8 个 output patches，每个 64 bf16，shape 是 `[8, 64]`；完整 Qwen3-0.6B `o_proj` 需要 2 个这样的 blocks。
- dataflow：每个 output patch 用两个 projection workers 分别处理低/高 32 output rows；每个 worker 一次接收 32-row/full-K packed Q4 buffer，在 core 内串行消费 8 个 256-wide K chunks 并直接输出自己的 32 rows。16 个 projection workers 覆盖一个 512-wide block，不需要 reducer。
- 数值：8-patch smoke 相对 CPU reference 的 `max_abs_error=0.0234375`。
- 性能：grouped-DMA 8-patch smoke 的 3 次样本 median 约 0.64 ms。
- 已确认的限制：单 worker 直接处理 64 rows x 2048 K 会 runtime timeout；把同一个 64-row patch 改成两个 32-row/full-K workers 后稳定运行。这和 FastFlowLM 的 “每列 4 个 compute rows 各吃一个 32-row Q4NX chunk” 强推断一致。
- 已解决的限制：最初 8-patch 512-wide block 会在自动 endpoint 放置时报
  `SequentialPlacer` 或 shim output DMA 超限；修复方式是显式绑定 runtime
  endpoint，并把 activation 从 c0 单独输入，避免与 c2..c5 的 weight ingress
  争用 shim output DMA。
- 集成验证：`QwenChunkedAttentionCurrent -> Q4NXFusedLinearProjection` 的两段
  smoke 已跑通，output-row/full-K 版本 `o_proj_max_abs_error=0.0244140625`，
  8-patch grouped-DMA 版本 3 次样本 median 约 1.45 ms。它验证 attention context 可以作为后续 online-Q4 projection 的
  输入继续消费。
- 失败实验：把 `Q4NXFusedQCurrentProjection -> QwenChunkedAttentionCurrent ->
  Q4NXFusedLinearProjection` 直接作为三段自动 fused runlist，会出现 runtime
  timeout；使用 4 个独立 `o_proj` block 时可跑完但 isolated `o_proj` error
  约 0.59。这个结果说明当前自动 operator 拼接已经不是可靠的 layer engine
  形式，下一步需要手工设计 layer-local dataflow，而不是继续增加 runlist 段数。

`q4nx_fused_linear_residual_projection` 当前是 `o_proj -> residual add` 的 layer-local phase 原型：

- 输入 activation：2048 bf16，对应 Qwen3-0.6B attention context。
- 输入 residual block：512 bf16，对应同一个 8-patch output block 的 residual slice。
- 输入 `o_proj` packed Q4NX stream：每个 output patch 81920 bytes，按 patch-major、K chunk 顺序保存。
- 输出：8 个 output patches，每个 64 bf16，shape 是 `[8, 64]`。
- dataflow：复用完整 512-wide `o_proj` fabric；projection worker 输出 32-row chunk，row1 join 成 64-row projected patch；edge residual workers 消费 projected patch 和共享 residual block，输出 residual-updated patch。
- 数值：8-patch smoke 相对 CPU reference 的 `max_abs_error=0.03125`。
- 性能：3 次样本 median 约 0.90 ms。
- 意义：这是第一个把 projection 后续 math 合进同一个 ELF 的边界，证明 `o_proj` output 不必回到 Python 做 residual add。下一步应把 attention output 直接接到这个 residual boundary，并继续减少中间 L3 drain。
- 集成验证：`QwenChunkedAttentionCurrent -> Q4NXFusedLinearResidualProjection`
  已跑通，覆盖 `attention -> O projection -> residual` 的 512-wide block；
  `o_proj_residual_max_abs_error=0.0244140625`，3 次样本 median 约 2.05 ms。
- 完整 hidden-size 验证：同一 fused call 串接两个 512-wide blocks，输出
  `[16, 64]`；`o_proj_residual_max_abs_error=0.0244140625`，3 次样本 median
  约 2.35 ms。

`qwen_chunked_attention_current` 当前是 decode attention 的 layer-local dataflow 原型：

- 输入 `q_current`：按 KV group 分组，包含该 group 的两个 Q head 以及当前 K/V。
- 输入 packet cache：沿现有 `K chunk, V chunk, mask chunk` layout 扫描。
- 当前 slot：kernel 在当前 chunk 内直接使用 `q_current` 里的 current K/V，不从 packet 读取该 row。
- packet 持久化：`QwenCurrentKVCacheWrite` 从同一个 `q_current` 中取 current K/V，
  只写当前 slot 的 K/V row 和 mask。mask DMA 必须 4 字节对齐，所以 writer 按两个
  bf16 mask 元素写入；current row 为 0 时写 `[1,0]`，否则写 `[1,1]` 覆盖 previous/current mask。
- 默认 Qwen3-0.6B 形状：8 KV groups、每组 2 个 Q heads、`head_dim=128`、`chunk_size=64`。
- 数值：8-group attention+writer smoke 相对 CPU reference 的 `max_abs_error=0.033203125`，packet update max error 为 0。
- 性能：`attend_seq_len=128`、3 次样本 median 约 1.32 ms，短 run 抖动较大；attention-only 5 次样本 median 约 0.77 ms。
- 已确认的限制：worker 不能有 `query + current_kv + packet` 三路输入，会报 input DMA channel exceeded；把 current row 写回为完整 updated packet chunk 会让 tile 同时保留 input chunk 和 output chunk，SRAM 超限。因此 current K/V 必须和 Q 合并成一路 stream，packet 持久化必须是独立小写入路径或后续 layer engine 内专用 writer。
- 已确认的失败路径：把 packet 小写回直接合入当前 attention operator，在 8-group
  形状下会遇到自动 placement 或 tile-local resource 限制。后续优化点不是继续
  拆更多 FusedMLIROperator 子图，而是做单层 fused layer 的手工 dataflow/placement：
  attention row2 负责 softmax/value accumulation，row3 或 memtile 专门处理
  current K/V update，且 q_current 不能通过额外 DDR fill 复制。

`qwen_plane_attention_current` 当前是 FastFlowLM 四平面 KV layout 的 attention reader 原型：

- 输入 `q_current`：按 KV group 分组，包含两个 Q heads、current K、current V。
- 输入 KV plane：四个物理 plane 顺序是 `k03, v03, k47, v47`，每个 token row 是 `4 KV heads x head_dim`。
- dataflow：worker 每次 acquire 一个 K/V group tile，循环执行 running softmax update；当前 slot 从 `q_current` 旁路，非当前 row 从 plane 读取。
- 数值：64-token/16-token tile smoke 相对 CPU reference 的 `max_abs_error=0.03125`；128-token/32-token tile smoke 的 `max_abs_error=0.0234375`；7-token tail smoke 的 `max_abs_error=0.046875`。
- 性能：64-token/16-token tile 的 3 次样本 median 约 0.79 ms；128-token/32-token tile 的单次样本约 0.92 ms，短 run 抖动明显。
- 当前高层实现为每个 plane pair、每个 chunk 发一个 ordinary runtime fill；64-token/16-token tile 和 128-token/32-token tile 都是 8 个 KV fill tasks，不再按 8 个 KV group 重复读。
- 集成验证：`Q4NXFusedQCurrentProjection -> QwenPlaneAttentionCurrent` 已跑通，覆盖 packed-Q4 q_current projection 到四平面 attention 的直接替换边界。64-token/16-token tile 的 `attention_max_abs_error=0.015380859375`，3 次样本 median 约 1.89 ms；128-token/32-token tile 的 `attention_max_abs_error=0.007198333740234375`。
- 已确认的限制：128-token/16-token tile 仍会生成 16 个 KV fill tasks，加上 Q/drain 后触发 BD 预算；最终长 context reader 仍应把 per-chunk runtime tasks 收敛成 row1 memtile static BD chain + lock ring。

`qwen_qkv_to_q_current` 当前是 QKV projection 与 attention 之间的 fused-layout bridge：

- 输入：`Q4NXFusedQKVProjection` 的 patch 输出 `[patch, Q/K/V, 64]`。
- 输出：按 KV group 分组的 `q_current = [Q heads, current K, current V]`。
- 当前默认边界：8 个 output patches 只能完整覆盖前 2 个 KV groups，因为每个 group 需要 4 个 Q patches、2 个 K patches、2 个 V patches。
- 集成验证：`QKV projection -> QwenQKVToQCurrent -> QwenChunkedAttentionCurrent -> QwenCurrentKVCacheWrite` 已在同一个 fused ELF 中跑通；2-group smoke 的 `q_current_max_abs_error=0.0078125`、`attention_max_abs_error=0.0068359375`、packet update max error 为 0，3 次样本 median 约 1.71 ms。
- 这个 bridge 现在主要保留为旧 QKV patch surface 的验证边界；正式 decode-attention 路线改用 `Q4NXFusedQCurrentProjection` 的 group-major stream 覆盖全部 8 个 KV groups，避免计算暂时不用的 K/V patches。

## 验收标准

每一步合入正式代码前必须同时满足：

- 有端到端或稳定边界的集成验证，不只看单 op 数字。
- Q4 权重不在 host 热路径 dequant 成 BF16 BO。
- operator profile 能说明权重读、dequant、accumulate、输出各自耗时。
- event trace 能回答 DMA 和 compute 是否 overlap。
- batch=1 decode 的 column/core 利用率有改善。
- context 增大时 attention latency 按有效 KV tile 增长，而不是扫空 cache。

最终目标不是“更少 ELF 文件”，而是更接近高性能 NPU dataflow：

- packed Q4 权重在线消费。
- hidden stream 在 layer 内复用。
- Q/K/V 和 up/gate projection fusion。
- KV cache 直接按 attention layout 读写。
- runlist 降低 host dispatch。
- `quantized_qwen3` 和 `exported_llama3` 共享同一套 layer-engine 生成和 operator 主线。

## 非目标

- 不继续以跨所有 layers 的巨型 ELF 作为首要目标。
- 不把 batch padding 当作 batch=1 利用率优化。
- 不保留多套量化/权重格式 fallback。
- 不让 Python/Torch 操作留在 decode 热路径。
- 不为 generic GEMM 增加越来越多特殊参数来模拟 layer engine。
- 不先优化 prefill 而推迟 decode layer fused online-Q4。
