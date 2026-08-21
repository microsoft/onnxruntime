# BFloat16 Support Status and Plan (CPU and CUDA Execution Providers)

This document analyzes the current state of `bfloat16` (BF16, ONNX `tensor(bfloat16)`)
support in the **CPU** and **CUDA** execution providers (EPs) and proposes a phased plan
to add comprehensive support. The motivation is that the majority of modern models
(LLMs, diffusion, and other transformer architectures) are now published in BF16, and
running them today often requires up-front conversion to FP32/FP16, which costs memory
bandwidth, accuracy, or both.

> The per-operator data in this document was derived from the registered kernels listed
> in [`docs/OperatorKernels.md`](./OperatorKernels.md). That file is auto-generated from
> the kernel registry, so it is the authoritative snapshot of what each EP actually
> supports. Re-run [`tools/python/gen_opkernel_doc.py`](../tools/python/gen_opkernel_doc.py)
> after adding kernels to refresh the counts.

## 1. Executive summary

| EP | Ops with BF16 / total | Coverage | Character of support |
|----|-----------------------|----------|----------------------|
| CPU (`ai.onnx`) | 45 / 197 | ~23% | **Data-movement only.** Cast, Reshape, Gather, Concat, Slice, Split, Transpose, Scatter, control-flow, etc. **No arithmetic / GEMM / normalization / activation compute kernels.** |
| CPU (`com.microsoft`) | 3 / 95 | ~3% | Only `ExpandDims`, `GatherND`, `UnfoldTensor` (all data-movement). |
| CUDA (`ai.onnx`) | 83 / 149 | ~56% | **Broad compute support:** elementwise math, `MatMul`/`Gemm`, `Conv`, `Softmax`, `LayerNormalization`/`RMSNormalization`, activations, reductions, plus all data-movement ops. |
| CUDA (`com.microsoft`) | 25 / 80 | ~31% | Transformer fusions: `Attention`, `MultiHeadAttention`, `GroupQueryAttention`, `PagedAttention`, `SparseAttention`, `SkipLayerNormalization`, `FastGelu`, `MatMulNBits`, `MoE`, etc. |

**Key takeaways**

- **CUDA** can already run most BF16 transformer/CNN graphs end-to-end. The gaps are a
  long tail of less-common ops, a handful of fusions, and quantization/IO ops.
- **CPU** has effectively **no BF16 compute**. A BF16 graph can be reshaped and gathered
  but cannot be multiplied, normalized, or activated. The element-wise kernels even carry
  explicit `// Supposed to add BFloat16 but we are not supporting now` markers
  (see `onnxruntime/core/providers/cpu/math/element_wise_ops.cc`). In practice every BF16
  model falls back to FP32 on CPU, which requires inserting `Cast` nodes and doubles the
  memory footprint.

## 2. Current support detail

### 2.1 CPU EP

BF16-enabled `ai.onnx` ops (all structural / data-movement, no compute):

```
BitCast, Cast, Compress, Concat, ConcatFromSequence, ConstantOfShape, DynamicSlice,
Flatten, Gather, GatherElements, GatherND, Identity, If, IsInf, IsNaN, Loop,
MemcpyFromHost, MemcpyToHost, Optional, OptionalGetElement, OptionalHasElement,
RandomNormalLike, RandomUniformLike, Reshape, ReverseSequence, Scan, Scatter,
ScatterElements, ScatterND, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase,
SequenceInsert, SequenceLength, Shape, Shrink, Sign, Slice, Split, SplitToSequence,
Squeeze, TensorScatter, Transpose, Unsqueeze
```

Notably **missing on CPU** (high-value, required by virtually every BF16 model):
`MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sqrt`, `Erf`, `Exp`,
`Softmax`, `LayerNormalization`, `SimplifiedLayerNormalization`, `RMSNormalization`,
`Gelu`, `Relu`, `Sigmoid`, `Tanh`, `Where`, `ReduceSum`/`ReduceMean`, `Conv`,
`RotaryEmbedding`, and all `com.microsoft` attention/normalization fusions.

### 2.2 CUDA EP

BF16-enabled `ai.onnx` compute ops include:

```
Abs, Add, Sub, Mul, Div, Pow, Neg, Min, Max, Sum, Mod, Exp, Sqrt, Erf, Cos, Sin,
Relu, Sigmoid, Tanh, HardSigmoid, HardSwish, Gelu, Softmax, ReduceSum,
LayerNormalization, SimplifiedLayerNormalization, RMSNormalization, RotaryEmbedding,
MatMul, Gemm, Conv, DeformConv, AveragePool, RoiAlign, Where, TopK, Tile, Trilu, Expand,
Dropout, Attention
```

plus all the data-movement ops the CPU EP has.

BF16-enabled `com.microsoft` (contrib) ops:

```
Attention, MultiHeadAttention, GroupQueryAttention, PagedAttention, SparseAttention,
SkipLayerNormalization, SkipSimplifiedLayerNormalization, BiasGelu, FastGelu, QuickGelu,
Gelu, RotaryEmbedding, FusedMatMul, TransposeMatMul, GemmFloat8, MatMulNBits, MatMulBnb4,
GatherBlockQuantized, MoE, QMoE, Trilu, UnfoldTensor, BiasDropout, BitmaskDropout,
BitmaskBiasDropout
```

Representative CUDA gaps (compute ops still FP16/FP32-only): `BatchNormalization`,
`InstanceNormalization`, `GroupNorm`/`SkipGroupNorm`, `MaxPool`/`GlobalAveragePool`,
`PRelu`, `LeakyRelu`, `Elu`, `Selu`, `Clip`, `Log`, `Reciprocal`, `Floor`/`Ceil`/`Round`,
`Resize`, `GridSample`, `CumSum`, `Einsum`, `Pad`, `ReduceMean`/`ReduceMax`/`ReduceMin`
(only `ReduceSum` is BF16), `Range`, `NonZero`, `OneHot`, `MoE` variants beyond the
registered ones, and several `QOrdered*`/quantization ops.

## 3. Cross-cutting infrastructure considerations

- **Type plumbing already exists.** `BFloat16` is a first-class element type
  (`include/onnxruntime/core/common/float16.h`), `Cast` supports BF16 on both EPs, and
  many template-based kernels are instantiated per dtype, so adding a registration is
  often the bulk of the work.
- **CPU has no native BF16 math path.** MLAS only exposes BF16 acceleration through
  `MlasSBGemm*` (ARM64 NEON `bf16` dot-product, used today as an *FP32 fastmath* path that
  internally converts FP32→BF16, see `onnxruntime/core/providers/cpu/math/matmul.cc`).
  There is no x86 BF16 GEMM kernel and no generic BF16 element-wise path. A comprehensive
  CPU plan therefore needs either (a) native BF16 kernels (AVX512-BF16 / AMX where
  available, NEON `bf16` elsewhere) or (b) an upcast-compute-downcast reference path for
  correctness and portability.
- **Accumulation precision.** BF16 has 8 mantissa bits; reductions, GEMM, normalization,
  and softmax must accumulate in FP32 to preserve accuracy (CUDA kernels already follow
  this convention). Any new CPU kernel must do the same.
- **Testing.** `OpTester` already supports `BFloat16` inputs/outputs; existing CUDA BF16
  tests are the template to follow. Tolerances must reflect BF16's reduced precision.

## 4. Proposed plan

The plan is ordered to deliver the highest model-coverage per unit of effort first.

### Phase 0 — Tooling and tracking (low effort)
- Add a coverage report (extend `gen_opkernel_doc.py` or a small script) that emits, per
  EP, the list of ops missing BF16, so progress is measurable and regressions are visible.
- Define BF16 accuracy/tolerance conventions and a shared test helper.

### Phase 1 — CPU: enable BF16 for the transformer core (highest impact)
Add BF16 kernels (FP32-accumulating reference implementations first, MLAS/SIMD
acceleration as a follow-up) for the ops that gate end-to-end BF16 execution:
1. `MatMul`, `Gemm`
2. Element-wise binary/unary: `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sqrt`, `Exp`, `Erf`,
   `Neg`, `Abs`, plus the explicitly-deferred `Sum`/`Max`/`Min`
3. Activations: `Gelu`, `Relu`, `Sigmoid`, `Tanh`, `Softmax`
4. Normalization: `LayerNormalization`, `SimplifiedLayerNormalization`, `RMSNormalization`,
   `SkipLayerNormalization`
5. `Where`, `ReduceSum`/`ReduceMean`, `RotaryEmbedding`

This unblocks running BF16 LLMs on CPU without inserting `Cast` nodes.

### Phase 2 — CPU: convolutional / vision and remaining elementwise
`Conv`, `BatchNormalization`, `InstanceNormalization`, pooling (`MaxPool`, `AveragePool`,
`GlobalAveragePool`), `Resize`, `Clip`,
`PRelu`/`LeakyRelu`/`Elu`, remaining reductions and unary math.

### Phase 3 — CPU: native BF16 acceleration
Add hardware-accelerated kernels behind CPU feature detection
(`MlasBf16AccelerationSupported`): AVX512-BF16 / AMX on x86, NEON `bf16` on ARM64, with
the Phase 1/2 reference kernels as the portable fallback.

### Phase 4 — CUDA: close the long tail
Add BF16 registrations/kernels for the remaining compute ops (normalization/pooling/
activation/reduction list in §2.2) and the missing contrib fusions, prioritized by how
often they appear in published BF16 graphs. Many CUDA kernels are already templated, so
this is frequently registration-only plus a test.

### Phase 5 — Validation and documentation
- End-to-end BF16 model tests on representative LLM/CNN graphs for both EPs.
- Refresh `docs/OperatorKernels.md` and this document; update the coverage report.

## 5. Acceptance criteria

- A representative BF16 transformer model runs end-to-end on the CPU EP with **no inserted
  `Cast`-to-FP32** for the core compute path (Phase 1).
- CUDA BF16 op coverage reaches parity with FP16 for the common transformer/CNN op set
  (Phase 4).
- Each newly added kernel ships with a BF16 unit test using agreed tolerances, and the
  generated operator docs are regenerated.
