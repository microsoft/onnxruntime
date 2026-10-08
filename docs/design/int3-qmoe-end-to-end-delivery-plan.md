# INT3 QMoE End-to-End Delivery Plan

## 摘要与文档状态

日期：2026-10-08。本文沿用 [ORT roadmap PR #32657](https://github.com/microsoft/onnxruntime/pull/32657) 和 [INT2 端到端交付计划](int2-qmoe-end-to-end-delivery-plan.md) 的阶段门方法，遵守 [低位宽探索文档](2bit-6bit-weight-only-quantization-exploration.md) 的 canonical bitstream / provider prepack 边界，定义均匀 INT3 QMoE 的基础版本、后续性能路径，以及准备提交 ONNX committee 讨论的可移植格式草案。

**这是设计提案，不是已经批准的 ONNX 标准，也不是现有 QMoE 已支持 INT3 的声明。** 当前 `com.microsoft::QMoE` 是 ORT contrib operator。新增位宽、标准域算子或 tensor datatype 必须分别走 schema review、版本管理和 ONNX 审议流程；本计划不自行分配 ONNX INT3 datatype 编号或标准 opset。

先交付可量化、可导出、可加载、可执行 decode 和 prefill 的可信基础版，再决定优化投入。基础版不承诺快于 INT2/INT4；生产性能资格必须单独验收。INT3 的价值首先是验证质量、有效模型大小和运行成本之间是否存在实用折中。

```text
INT3 可移植契约与参考字节向量
  -> 独立 scalar oracle 与共享验证
  -> 有界 CUDA 正确性 decode + prefill
  -> packed CUDA decode
  -> 有界 selected-expert prefill / 原生 packed prefill
  -> CPU reference 和跨 provider parity
  -> Olive 量化 + Mobius fused QMoE 导出
  -> Qwen 模型质量、容量和性能验收
  -> ONNX committee 提案材料与互操作评审
```

导出、CPU reference 和 CUDA 可在契约冻结后并行。已有 INT2 经验可以复用，但不能把其类型、整除公式或 runtime prepack 字节直接当作 INT3 实现。

## 1. 产品目标与初始范围

初始目标为 Qwen3-30B-A3B 类模型：FC1 gate/up 使用 INT3，FC2 down 保持 INT4，敏感非专家参数使用已有较高精度方案。先构造小型确定性 fused fixture，再做完整模型，不能从全尺寸模型才开始排查字节错误。

| 项目 | 基础版要求 | 后续研究，不阻塞基础版 |
| --- | --- | --- |
| 位宽组合 | `(FC1, FC2)=(3,4)`，fused FC3 与 FC1 同位宽 | `(3,3)`、`(4,3)` 与其他混合策略 |
| 激活 | FP16/BF16，FP32 accumulation，分别验证 | INT8 activation、DP4A、INT8 Tensor Core |
| 权重量化 | 均匀、对称、按 K 分组，首个模型 block64 | 非对称、码本、额外 block size |
| 逻辑 block size | 格式 profile 定义 32/64/128；首个优化 kernel 可只支持 64 | 根据证据扩展优化分派 |
| 融合 | `swiglu_fusion=1`，明确 gate/up 行交错 | 非融合 FC3、concat 布局，须单独资格验证 |
| routing | 多专家、top-k、单/多 token、专家空桶 | 并发、capture 等生产覆盖仍需阶段验收 |
| 平台 | CUDA SM80/A100 为首次实机目标，CPU 为参考 | SM86/89、H200、Spark、其他 provider |
| prefill | 基础版必须可运行且 scratch 有界 | 原生 packed grouped GEMM 是性能阶段 |

不改变 routing、归一化、activation 参数和 FC2 的既有语义；bias 覆盖应加入正确性 fixture，不把无 bias 的优化资格误写为算子格式限制。不导入 IQ3/Q3_K 字节，不扩大为完整 dense `MatMulNBits(bits=3)` 产品承诺。

## 2. 面向 ONNX 的 INT3 可移植格式草案

以下 MUST/SHOULD 描述**本提案内部**的规范要求。委员会尚未认可；在 schema 和共享向量冻结前，导出器不得将其声称为稳定通用标准。

### 2.1 数学语义与码值

每个权重的存储 code 为无符号整数 `u in [0,7]`。首版对称 profile 的隐式 zero point 为 `z=4`，逻辑有符号值为 `q=u-4 in [-4,3]`，不是 three-bit two's-complement。

对于逻辑权重 `W[e,n,k]`：

```text
b = floor(k / block_size)
W_dequant[e,n,k] = (u[e,n,k] - 4) * S[e,n,b]
```

`S` MUST 为有限、非负数；`S=0` 表示该组所有解码权重为零。exporter SHOULD 将这种组的 code 置为 4，以得到确定性零组。解码语义不依赖 RTN、GPTQ 或校准方法。

独立 RTN fixture 的默认量化规则为：

```text
s = max(max(-W, 0) / 4, max(W, 0) / 3)  # 对每个实际 K block 取最大值
q = clamp(round_to_nearest_even(W / s), -4, 3)
u = q + 4
```

全零组直接输出 `s=0,u=4`。fixture 先将 scale 转为最终存储 dtype，再用该已舍入 scale 生成 code，若 scale 下溢为零则按零组处理；记录误差，不默默改变 dtype。真实量化工具可采用其他 scale 优化算法，但 MUST 记录方法、舍入、clipping 和 scale dtype。量化算法不是 packed 解码标准的一部分。

### 2.2 唯一的模型序列化布局：逐行 LSB-first 连续位流

可移植模型 MUST 使用 `uint8` tensor 容器；不要求新的 ONNX INT3 tensor datatype。每个专家的每个输出行分别打包，K 方向连续：

ORT 首个 exporter MUST 显式设置 `quant_type='int'`、`weights_prepacked=0`、`block_size` 和有效 FC 位宽，不能依赖当前 provider-specific prepacked 默认行为。`H` 由 activation 的 hidden dimension 得到，`I` 由 fused FC1 的逻辑输出行数除以 2 得到；再验证 FC2 逻辑行数和 packed K 维，不能从 packed byte 数反推 I。

```text
logical shape: [E, N, K]
row_bytes R = ceil(3 * K / 8)
packed shape: [E, N, R]
row byte offset = (e * N + n) * R
value k starts at bit offset 3 * k within that row
```

每个 code 的最低有效位先写入；位流按 byte 地址递增，byte 内按 bit0 到 bit7 递增。value 可以跨字节，禁止每个 INT3 值补成 nibble。行与行、专家与专家之间不共享字节；最后一个 byte 的未使用高位 MUST 为零。

block 是 scale 的逻辑分组，不在位流中插入 header 或额外 padding。对于 profile 的 block32/64/128，完整 block 自然字节对齐，但 decoder MUST 依据上述位偏移定义，而不是假设 `8 / bits` 个整数能整除一个 byte。

模型字节是唯一真值，不依赖 CPU 大小端。跨字节读取的参考式为：

```text
bit = 3 * k
byte = floor(bit / 8)
shift = bit % 8
word = P[byte]
if byte + 1 < R: word |= P[byte + 1] << 8
u = (word >> shift) & 7
```

读取 MUST 不越过本行末尾。所有大小、stride 和 offset 算法 MUST 做溢出检查。

### 2.3 逻辑尾部与物理尾位

格式允许 `K` 不是 block size 或 8 的整数倍，scale block 数 `B=ceil(K/block_size)`，最后一组仅包含剩余真实元素。`K` MUST 来自逻辑 shape/算子属性，不由 packed byte 数倒推。

未使用的 byte 高位是**物理 padding bits**，其零值不代表一个额外的 `q=-4` 权重。运行时若为了向量化增加 K padding，则 MUST mask 掉额外元素或使用逻辑零 code=4，并且不参与 routing、scale 分组或输出语义。不得将 runtime padding 重新序列化为原始 K。

首次 CUDA 优化可要求 `K % block_size == 0` 等对齐条件，但这些是执行资格，不是可移植格式限制。不能执行的合法模型应明确 unsupported 或选择有界 reference fallback，不能静默误读尾部。

当前 QMoE schema 要求 H/I 整除 block size 且权重末维 byte-aligned。本提案的 ceil/tail 语义不是当前 schema 能力；P1 必须评审该扩展。首个 ORT 模型保留 H/I 整除共享 block size 的约束，若尾部扩展延期，则 portable-format fixture 可以先定义尾部，但 ORT exporter 必须拒绝未获 schema 支持的尾部模型。

### 2.4 FC1/FC2 顺序与 scale shape

对 `E` 个专家、hidden `H`、intermediate `I`，首版 fused SwiGLU 的逻辑行顺序规定为：

```text
FC1 logical: [E, 2*I, H]
FC1 row 2*j     = gate[j,:]
FC1 row 2*j + 1 = up[j,:]
Y[j] = activation(gate[j]) * up[j]
FC1 INT3 bytes: [E, 2*I, ceil(3*H/8)]
FC1 scales:     [E, 2*I, ceil(H/block_size)]

FC2 logical: [E, H, I]  # down，输出行是 hidden，K 方向是 intermediate
FC2 INT4 bytes: [E, H, ceil(4*I/8)]
FC2 scales:     [E, H, ceil(I/block_size)]
```

scale 的 expert/output-row 顺序 MUST 与权重一致。可移植格式只允许 `float32/float16/bfloat16` scale，禁止 FP8 或整数 scale。首个 ORT 执行 profile 要求 FC1/FC2 scales 同 dtype 且与 FP16/BF16 activation 相同；FP32 scale 或独立 FC scale dtype 组合由后续 schema/provider 资格评审决定，未支持组合明确拒绝而非重新解释字节。基础 profile 不加入 row-wise scale 的额外 shape 隐式解释。

导出 fixture MUST 验证 gate/up 顺序和 down 的 transpose，不能只验证 packed 元素数。首个 exporter 采用 native fused QMoE 构造；从 dense 图自动识别并融合属于后续阶段。

### 2.5 Zero point、混合位宽与版本

首版 profile 只接受省略 zero point，或显式提供全部为 4 的 INT3 zero point。若提供，按每个 `[e,n]` 的 B 个 block code 独立 LSB-first 打包：`Z shape=[E,N,ceil(3*B/8)]`，最后 byte 未用高位为零。**不是**对整个 zero-point tensor 展平成一条无行边界位流。

每个 `[e,n]` 是一条包含 B 个 code 的连续位流：`Z_row_bytes=ceil(3*B/8)`，行 byte offset 为 `(e*N+n)*Z_row_bytes`，block b 的起始位为 `3*b`，复用第 2.2 节的跨 byte 提取规则。省略 Z 输入时 decoder 对每个 block 使用 4；显式输入时 validator 必须检查每个逻辑 code 等于 4，并拒绝非 4 值或非零尾位。两种表示的输出完全相同；默认 exporter 省略 Z 以节省空间，显式形式用于互操作和未来扩展验证。

非对称 zero point 的一般式是 `(u-z)*S`，但任意 `z` 属后续 profile/schema 评审，不在首版暗中接受。FC2 INT4 沿用已批准的 INT4 zero-point 语义，不把 INT3 的默认值 4 用于 INT4。

FC 位宽沿用独立 optional override 的设计：省略 `fc1/fc2/fc3_expert_weight_bits` 时继承 `expert_weight_bits`，不修改已有默认 4。fused FC3 与 FC1 相同，因为其数据位于 FC1 tensor。只在专门评审的 schema 修订后允许有效位宽 3；现有模型不得由于新增 INT3 被重新解释。

不使用 `pack_size=8/bits` 计算 INT3 shape/stride。通用规则是 `ceil(K*bits/8)`；已有 INT2/4/8 必须保持字节和数值兼容。格式改动若改变模型字节含义，MUST 有显式 schema/opset 版本或经批准的格式标识，不能仅靠 exporter 版本或 GPU 架构猜测。

### 2.6 Golden vectors 与可核验例子

```text
q:     [-4,-3,-2,-1,0,1,2,3]
u:     [ 0, 1, 2, 3,4,5,6,7]
bytes: [0x88, 0xc6, 0xfa]

K=3, q=[-4,0,3], u=[0,4,7]
bytes: [0xe0,0x01]  # 第二 byte 的 bit1..7 是零 padding

K=8 的全零逻辑权重，u 全为 4
bytes: [0x24,0x49,0x92]
```

必须提供机器可读 fixtures：逻辑 shape、位宽、block size、code、scale、zero point、预期 raw bytes、显式反量化权重和 QMoE 输出。覆盖所有 code、跨 byte、跨 block、多个行/专家、尾部、零组和截断/非法 padding。至少两份独立 pack/unpack 实现逐字节一致；不能用同一个有 bug 的 helper 同时生成 expected 和 actual。

## 3. CUDA 内部布局：2+1 位平面候选

**模型仍使用第 2 节连续位流。** CUDA 可以在 session 初始化时把它转换为低 2 位平面和高 1 位平面：

```text
lo2[k] = u[k] & 3
hi1[k] = (u[k] >> 2) & 1
u[k] = lo2[k] | (hi1[k] << 2)
q[k] = u[k] - 4
```

这是借鉴 llama.cpp `Q3_K` 低位/高位分离的物理思想，不采用其 hierarchical scale、量化算法或 GGUF ABI；IQ3 码本格式更不是均匀 INT3。位平面各自可 tile、interleave、对齐并有 provider descriptor，但 descriptor 必须区分位宽、layout version、架构、逻辑形状和 padding；cache key 包括改变语义的输入及版本信息。

具体 tile、byte alignment、planes offset 和融合布局由 kernel 原型测量后决定；不是委员会格式契约。不得把该内部缓存作为 `weights_prepacked=0` 导出，也不在首版承诺可移植 offline prepack。显存统计必须包括 raw weights 与 prepack cache 的共存及额外 padding，不能只报三位 payload。

优先 FP16/BF16 activation；packed decode 就地解码、恢复 scale 并 FP32 累加。DP4A/INT8 Tensor Core 需另行 activation 量化和精度验收，不应成为基础 INT3 格式依赖。A100/H200 没有这里所需的原生 packed INT3 运算；硬件 INT8 支持不等于无需解码或自动提速。

## 4. 工作流、PR 切分与验收门

所有阶段目前为 **Planned**。以下是依赖驱动的计划，不是已合入实现或对其他团队的交期承诺。

| 阶段 / 候选 PR | 主要交付物 | 退出条件 |
| --- | --- | --- |
| P0：规范和 feasibility | 审阅第 2 节；冻结 fixtures；核对 Olive 已有 INT3 checkpoint 格式与本规范的差异 | exporter/runtime owners 同意字节及数学语义；独立 pack/unpack 对所有 code、尾部、零组逐字节一致；不直接复用未知 checkpoint packing |
| P1：QMoE 契约和共享验证 | 经评审允许 INT3；ceil-bit size/stride；shape inference；尾部和 zero-point 验证 | `(3,4)` 模型能验证；老 INT2/4/8 fixtures 完全兼容；未支持 provider 清晰拒绝 |
| P2：独立参考与 CUDA 正确性 | scalar FP32 oracle；FP16/BF16 数值参考；有界 expert/block dequant；decode 和 prefill | synthetic/reduced Qwen 正确；无无界完整专家反量化；实际 provider target 构建测试通过 |
| P3：packed CUDA decode | 原始位流到 versioned prepack；就地 INT3 load/decode；fused FC1；保留 FC2 INT4 | 目标 routing/shape parity、memcheck；无大反量化 buffer；记录 prepack 与重复调用成本 |
| P4a：prefill 基础路径 | selected-expert 或 row/block tiled dequant + 现有 GEMM；明确 scratch budget | 长 prompt、partial tile、decode/prefill 转换正常；内存有界；禁止标为 native packed 性能 |
| P4b：prefill 性能路径 | packed grouped kernel 或 measured 局部转换策略；block64 先行 | 端到端成本优于基础路径；scratch 估计/边界回归通过；没有收益则停止 native 优化 |
| P5：Olive/Mobius | expert 分类、mixed checkpoint、packing converter、fused graph、external data、manifest | checkpoint→ONNX→ORT parity；位宽绑定和 gate/up 顺序正确；source revision/工具版本可复现 |
| P6：CPU parity | 基于相同 raw bytes 的 CPU reference 执行和 invalid-model 校验 | CPU/CUDA 和独立 oracle 在约定容差内；CPU 不是生产吞吐承诺 |
| P7：模型资格 | INT2/INT3/INT4 matched recipe 的质量、容量、prefill、decode | 满足预先约定的质量/容量目标；性能结论可重复；提供 fail/stop 决策 |
| P8：委员会材料 | 独立于 CUDA 的规范、reference、互操作 fixture、schema 版本提案 | ONNX 评审决定容器/operator/版本路径；不把 ORT 合并当作 ONNX 批准 |

P3/P4a 可以在 P2 后并行；P5/P6 在 P1 fixture 冻结后并行。若只有一个工程师，先 P0/P1/P2，再打通一个完整 Qwen 小模型导出，之后实现 packed decode 和基础 prefill；不要同时引入 activation INT8、INT3码本和多个 GPU 特化。

### 4.1 责任与排期方法

P0/P1 由 ORT operator owner 与 Olive/Mobius exporter owner 联合批准；P2/P3/P4 由 CUDA owner 负责；P6 由 CPU owner 负责；P7 由模型质量与 benchmark owner 复核；P8 由 ONNX proposal sponsor 协调。上述是待认领角色，不是已经指定的人或其他团队的承诺。

在 P0 结束时分别估计 schema/reference、CUDA 正确性、packed decode、基础 prefill、export、模型 qualification 的工时，再根据人员与实际 PR 审阅速度给日历日期；native prefill 和委员会审批单列，不写入基础版硬截止。每周只更新已关闭的 gate、当前阻塞及下一份可验证交付物。ONNX 标准化周期不应阻塞 ORT contrib 设计实验，也不能因 ORT 交付而被宣称完成。

### 4.2 基础版与生产版必须分开命名

基础版完成：原始 checkpoint 可量化并导出、fixture/CPU/CUDA parity、模型 decode/prefill 可执行且 scratch 有界、至少一组完整质量与容量/速度基线。不能只有 kernel unit test，不能把 dequant correctness fallback 当作性能交付。

生产资格完成：实际 deployment shape 的 packed 路径与分派被证明，prefill/decode/memory 没有未解释退化，重复测量与精度目标通过，fallback、并发、capture、scale 更新和 prepack lifetime 在目标部署方式下有验证。支持 SM80 的代码或 dispatch 条件不等于已验证所有 SM80+ 硬件。

## 5. 验证矩阵与性能方法

- 格式：code 0..7、跨 byte、K 尾部、block 尾部、多个专家/输出行、全零组、scale dtype、截断 external data、overflow、非法 shape/zero point/padding。
- 算子：FP16/BF16、top-k 1/2/8、单/多 token、空专家、bias、routing permutation、SwiGLU 参数、FC2 orientation 和旧模型回归；非法输入在计算前拒绝。
- kernel：测试 packed eligibility 和 fallback；真实 decode→prefill→decode 转换、partial row tile、cached/runtime scale、预处理复用、scratch guard、sanitizer。独立 harness 不能替代实际 legacy/plugin provider target；plugin 未测即明确标注未覆盖。
- 工具：先确认 Olive 的 native INT3 checkpoint 能力，再资格验证 fused expert 覆盖、Mobius graph/initializer/external data 绑定；支持 native INT3 linear 不等于支持本规范的 QMoE 导出。
- 模型：固定 checkpoint/tokenizer、tensor placement、block size 和非专家精度，对照 INT2/INT3/INT4；增加浮点质量参考。不用不同 block/不同量化层掩盖位宽效应。
- 性能：同 GPU、同 runtime、profiling 关闭、正反/交替 A/B；分别测 load/prepack、TTFT、prefill、decode TPS、显存峰值与 scratch；记录主机 orchestration 范围。profile 单独证明实际分派，不用于报告无 profile TPS。
- 精度：固定任务与样本、校准/评估 split 分离，报告 logits/层误差和完整任务指标；逐元素相同或 greedy token 相同都不能代替质量验收。

在开始全模型验收前冻结任务质量容忍值、有效大小收益和性能非退化标准；由产品与 exporter/runtime owners 共同确认，不能事后按结果移动阈值。小于测量波动的吞吐变化应报告不确定，而不是成功。

## 6. 有效存储成本与停止条件

对每个 `[E,N,K]` INT3 tensor，权重 payload 为 `E*N*ceil(3*K/8)` 字节，scale 为 `E*N*ceil(K/block_size)*sizeof(scale)`；显式 zero point、padding、manifest、未量化参数和 external-data alignment 另外计入。

例如 K 对齐、FP16 scale、无显式 zero point、block64 时，权重加 scale 的有效位率为 `3 + 16/64 = 3.25 bit/weight`，而不是恰好 3。与相同 scale 策略的 INT4 比较是 4.25 bit/weight，不能把名义 payload 的 25% 节省直接写为整模型或显存节省。

停止/调整条件：质量无明显优于 INT2 的价值；有效容量无足够优于 INT4 的优势；prepack/raw cache 或临时展开吃掉容量收益；packed extraction 的成本抵消带宽收益；只在微基准快而完整模型退化；或 exporter/runtime 字节解释无法互操作。没有实测收益时保留基础正确性和结果，不为展示新位宽而加入不可维护优化特判。

## 7. ONNX committee 提交包与开放决策

提交材料应包括：独立 packed-format 规范与数学语义、包含尾部/zero-point 的 golden bytes、shape inference 和错误模型定义、至少两个独立 decoder、跨 exporter/runtime 的互操作报告、标准图分解参考，以及真实性能/质量/容量证据。

推荐先用 `uint8` 容器明确 operator 的 opaque packed input 语义，复用现有 ONNX external data；是否新增 native INT3/UINT3 datatype 是单独决策。三位位流不能直接交给现有 `DequantizeLinear` 作为普通 uint8 元素；需要显式解码的参考分解或经批准的 packed operator。是否标准化整个 fused MoE 算子，还是先标准化通用 packed 量化表示/运算，也须单独讨论。

待审批问题：schema 升级是否需要新 contrib version；是否接受首版对称-only profile、32/64/128 block 集合和尾部规则；显式 zero-point 的扩展策略；scale dtype 的约束；规范名称/标识；参考分解与 shape 推导；旧 exporter 和不支持 provider 的诊断行为。评审解决前不宣称冻结了 ONNX 标准。

## 8. 首轮执行清单

1. 审阅并冻结可移植 raw 位流、码值/scale/zero-point 语义和 FC1/FC2 golden fixtures。
2. 核对现有 Olive INT3 checkpoint，增加到 canonical 位流的显式转换与 shared-fixture 资格测试。
3. 在独立参考与 schema 验证通过后，实现有界 CUDA 正确性，先打通小型 `(3,4)` decode/prefill 导出闭环。
4. 实现 FP16/BF16 激活的 packed decode 和基础 prefill，再测 Qwen INT2/3/4 的质量、容量和速度。
5. 根据结果选择 native prefill、其他 GPU 或整数激活研究；基础版无需等待这些实验。

本计划是文档设计，未新运行 INT3 kernel、模型或 CI。后续每个实现 PR 必须记录实际源码版本、构建配置、provider、硬件、测试结果和未覆盖范围，不能沿用 INT2 或本地原型结果作为 INT3 资格证据。