# DynamicQuantizeMatMulNBits - CUDA Operator Design

**Status:** Implementation in progress: schema, generic DP4A path, native
INT8 tensor-core dispatch with direct INT8/INT4/INT2 decoding, all three
activation granularities, and explicit default-quantizer export opt-in.
Aligned native paths use asynchronous packed-weight/activation staging:
two stages for single-interval W8 and three for grouped weights.
Hopper additionally has an integrated strict FP16 W8 TMA/WGMMA fast path
for eligible constant single-block weights.
Further crossover tuning and full cross-platform/model-level
performance acceptance remain follow-up work.
This is not yet the completed version-1 delivery described below.

**Motivation:** [ONNX Runtime issue #33200](https://github.com/microsoft/onnxruntime/issues/33200)
requests CUDA GEMM with dynamically quantized per-token INT8 activations,
per-channel INT8 weights, and a floating-point output with scaling fused into
the GEMM epilogue.

The proposed operator follows the execution architecture of
[MatMulBlockQuantizedFp4Weight](./matmul_block_scaled_fp4.md) and
[MatMulBlockQuantizedFp8Weight](./matmul_block_scaled_fp8.md), including optional
bias and separation of operator semantics from CUDA dispatch. Its serialized
weights and scales instead reuse the standard symmetric
[MatMulNBits](./matmul_nbits.md) representation so offline quantization tools
can produce the same initializers for both operators. Activations are always dynamically
quantized to symmetric INT8 inside the operator. Weight-only execution
remains the responsibility of MatMulNBits and the existing FP4/FP8 operators;
every backend must preserve W8A8, W4A8, or W2A8 integer-dot-product semantics.
Version 1 proposes INT8, INT4, and INT2 logical weights stored in MatMulNBits
UINT8 block blobs, with the same activation modes and floating-point
block-scale contract. `NBits`
refers to weight precision; activation precision remains INT8.

## 1. Goals and Non-Goals

Goals:

- Accelerate prefill and encoder GEMMs with symmetric per-token activation
  quantization and per-channel weight quantization.
- Support per-tensor, per-token, and per-token K-block dynamic activation
  quantization through one explicit model-level granularity attribute.
- Generalize the weight representation to scales that vary along K.
- Reuse symmetric MatMulNBits INT8/INT4/INT2 weight bytes and scales without
  requantization, while keeping activation quantization explicit.
- Support FP16 and BF16 activations, bias, and output.
- Fuse scaling and bias into the native GEMM output stage, avoiding a global
  INT32 output intermediate on the primary fast path.
- Provide correct small-M and general CUDA paths without changing the
  model's selected activation-quantization mode.
- Keep weights and the serialized operator contract architecture independent.
- Keep a weight-bit-width attribute so further integer widths can be added
  without renaming the operator or reinterpreting existing models.

Non-goals for the initial implementation:

- UINT8 activation quantization, asymmetric weight quantization, explicit
  zero points, or arbitrary group indices. Stored weights use MatMulNBits
  unsigned codes centered at the fixed symmetric midpoint.
- Reusing `weight_prepacked=2` or arbitrary architecture-specific layouts.
  The only prepacked exception is SM80 `weight_prepacked=1` for the explicitly
  gated SM80+ W2 block-64, W4 block-32, and W8 block-32 paths.
- Weight-only W8A16/W4A16/W2A16 execution, static activation scales, or externally
  quantized activation inputs.
- FP32 activation/output support or quantized output.
- Batched weight matrices or general transpose attributes.
- Automatically enabling W8A8/W4A8/W2A8 for ordinary floating-point MatMul nodes.
- Changing existing FP4, FP8, or MatMulNBits operator behavior.

SmoothQuant or other outlier-management transformations belong in model
preparation, not in an implicit runtime heuristic.

### 1.1 Activation-granularity priorities

For logical `A[M, K]`, distinguish per-token scales `[M]` from true
per-input-channel scales `[K]`. The latter reduce across M for each K
feature and are not part of version 1. Names such as "per-channel GEMM" in
other implementations often refer to weights, not activations.

| Priority | Activation mode | Rationale |
|----------|-----------------|-----------|
| P0 | `dynamic_per_token` | Default; independent of unrelated tokens, with a row-wise reduction and established W8A8/W4A8 usage. |
| P1 | `dynamic_per_block` | Localizes outliers within a token; important for block-scaled INT2/INT4/INT8 weights and producer-side quantization fusion. |
| P2 | `dynamic_per_tensor` | Explicit alternative with one scale, but batch-dependent accuracy and a tensor-wide reduction that may need extra launches. |
| Deferred | True dynamic per-input-channel activation quantization | Different reduction domain and K-varying scales; add only for a concrete model requirement, not as an alias for per-token. |

These are optimization priorities, not permission to omit correctness
support for schema-advertised modes. All three weight widths and all three
activation modes require working generic paths before version 1 is
advertised as complete. At M = 1, per-tensor and per-token have the same
reduction domain. Fewer scale values alone do not establish lower latency.

## 2. Operator Schema

Proposed name: `com.microsoft::DynamicQuantizeMatMulNBits`, version 1.
The proposed schema supports `bits = 8`, `bits = 4`, and `bits = 2`.

| Input | Index | Type | Shape and meaning |
|-------|-------|------|-------------------|
| `A` | 0 | FP16 or BF16 (`T`) | Row-major activation, `[..., K]`, rank at least 1. |
| `B` | 1 | UINT8 (`T1`) | Standard MatMulNBits blobs when `weight_prepacked=0`, or the same-sized SM80 fpA_intB buffer for the supported `weight_prepacked=1` path; shape `[N, G, blob_size]`, where `G = ceil(K / block_size)` and `blob_size = block_size * bits / 8`. |
| `scales` | 2 | FP16, BF16, or FP32 (`T2`) | MatMulNBits weight scales, `[N, G]` or flattened `[N * G]`, in output-channel/block order. |
| `bias` | 3 | Optional `T` | Per-output-channel bias, `[N]`. |

| Attribute | Type | Default | Meaning |
|-----------|------|---------|---------|
| `bits` | INT | `8` | Integer weight bit width; `8`, `4`, and `2` are supported by the proposed contract. Other widths are rejected. |
| `block_size` | INT | `128` | Number of logical K elements sharing a weight scale; power of two and at least 16, as in MatMulNBits. For `bits = 2`, at most 256. |
| `weight_prepacked` | INT | `0` | `0` uses standard blobs. `1` selects the SM80 fpA_intB layout and requires CUDA SM80+, W2 block 64 or W4/W8 block 32, K divisible by 128, N divisible by 64, and one activation-scale interval. |
| `activation_quantization` | STRING | `"dynamic_per_token"` | `"dynamic_per_tensor"`, `"dynamic_per_token"`, or `"dynamic_per_block"`. All modes dynamically quantize A to symmetric INT8. |
| `activation_block_size` | INT | No default | Required and positive for `"dynamic_per_block"`; must be absent for the other modes. |

Output `Y` has type `T` and shape `[..., N]`. Bias must have the same type as A.
Scales are dequantization multipliers, not their reciprocals. Their storage
type is independent of A so FP16/BF16 MatMulNBits initializers can be reused
unchanged while FP32 scales remain supported. Convert scale values to FP32
before the numerical operations in section 3.

The stored weight zero point is the fixed MatMulNBits symmetric midpoint,
`2 ** (bits - 1)`: 128 for INT8, 8 for INT4, and 2 for INT2. Subtracting it produces
logical signed codes `[-128, 127]`, `[-8, 7]`, and `[-2, 1]`, respectively; their logical
zero point is zero. There is no explicit `zero_points` input.
Generated activation codes use the symmetric range `[-127, 127]`.

The schema deliberately omits an `a_scale` input. Dynamically generated activation
scales are internal to the operator; optional-input presence must not
implicitly select a different quantization mode.
There is no `"none"` mode. `DynamicQuantize` makes mandatory activation
quantization explicit, while `MatMulNBits` identifies the extensible integer
weight representation. It is distinct from the weight-only MatMulNBits operator.

### 2.1 Per-channel weights are a single K block

For `K > 0`, choose a MatMulNBits-valid `block_size >= K` to obtain one
scale per output channel. Use `block_size = K` only when K is itself a
power of two and at least 16; otherwise choose a larger valid block size.
INT2 inherits MatMulNBits' maximum weight block size of 256, so a
single-block per-channel representation is possible only for `K <= 256`.
For larger K, INT2 uses multiple weight blocks; do not bypass that limit
with a sentinel or silently collapse their scales.

```text
A:       [M, K]      FP16/BF16
B:       [N, 1, block_size * bits / 8] UINT8
scales:  [N, 1] or [N] FP16/BF16/FP32
bias:    [N]         FP16/BF16, optional
Y:       [M, N]      FP16/BF16
```

Any valid block size with `G == 1` represents per-channel weights.
It is eligible for the final-epilogue-only fast path when the selected
activation mode also has no scale changes along K.
K itself need not be a power of two; the physical block includes unused
padding when `block_size > K`. Using this single-block representation avoids
a new sentinel value or a separate weight-granularity attribute.

Accept both `[N, 1]` and flattened `[N]` scales without changing their
values, matching MatMulNBits input validation. Padding overhead is part of
the storage/latency comparison for per-channel weights.

### 2.2 Block-scaled weights

For a MatMulNBits-valid weight block size S:

```text
G = K / S + (K % S != 0)
g(k) = k / S
B_scaled[n,k] = float(qB[n,k]) * float32(scales[n,g(k)])
```

`qB` is the centered logical signed weight code defined in section 2.4.
`scales[n,g]` denotes the same value for 2D or flattened scale storage. Block sizes
count logical K elements, not packed bytes. An activation scale boundary
can fall between packed codes in a byte; each code uses its own logical K index.
The last logical block may be shorter than S, but serialized B includes a
full MatMulNBits blob for it. Ignore codes beyond logical K. Additional
backend padding is internal and must not affect reductions.

### 2.3 Shape inference and validation

Derive N from B's first dimension and logical K from A's last dimension;
do not duplicate them in attributes. Propagate A's
element type, preserve all its leading dimensions, and replace its last
dimension with N. A rank-1 input `[K]` produces `[N]`.

Schema inference and runtime validation must check all available information:

- A has rank at least 1 and B has rank 3.
- `bits` is 8, 4, or 2. Unsupported widths produce an
  explicit error, not reinterpretation as another supported width.
- B has type UINT8 for all three widths and shape `[N, G, blob_size]`.
  Reject signed INT8 tensors and alternative linear/packed-code layouts.
- `block_size >= 16` is a power of two and the activation mode is recognized.
- For `bits = 2`, `block_size <= 256`, matching MatMulNBits validation;
  allowed weight block sizes are 16, 32, 64, 128, and 256.
- `activation_block_size > 0` is present exactly when the mode is
  `"dynamic_per_block"`.
- `scales` has shape `[N, G]` or `[N * G]` and a supported floating type;
  no scalar broadcasting or arbitrary equal-element-count shapes.
- Bias, when present, has shape `[N]` and type T.
- Dimensions and scratch sizes fit the backend's index and allocation types.

Unknown or symbolic dimensions must remain valid during shape inference.
B's padded block count does not determine exact logical K for any supported
width; do not infer K from its physical byte count. Validate against A
when logical K becomes known.
Use checked dimension and byte-size arithmetic at runtime, including padded
dimensions and rounding to backend alignments.

### 2.4 Symmetric MatMulNBits weight storage

Keep activation modes, activation-scale rules, weight block scales, bias,
and output semantics independent of `bits`. Existing models that omit
`bits` continue to use 8-bit weights.

| `bits` | B type | Bytes per block blob | Stored midpoint | Logical signed code range |
|--------|--------|----------------------|-----------------|---------------------------|
| 8 | UINT8 | `block_size` | 128 | `[-128, 127]` |
| 4 | UINT8 | `block_size / 2` | 8 | `[-8, 7]` |
| 2 | UINT8 | `block_size / 4` | 2 | `[-2, 1]` |

Within each block, INT4 uses the same low-nibble-first unsigned packing as
MatMulNBits. INT2 packs four codes per byte, at bit offsets 0, 2, 4, and 6
in increasing K order. Decode all three widths by subtracting the symmetric midpoint:

```text
g = k / block_size
j = k % block_size
pack = 8 / bits
byte = B[n,g,j / pack]
u = (byte >> (bits * (j % pack))) & ((1 << bits) - 1)
qB[n,k] = int32(u) - (1 << (bits - 1))
```

Division and indexing here are integer operations and `0 <= k < K`.
Packing restarts at every output-channel/block blob. For example, INT4
logical codes `[-8, 7, -1]` with `block_size = 16` occupy one eight-byte
blob beginning `[0xf0, 0x87]`, followed by six `0x88` bytes when padding is
encoded as logical zero. These are offset codes, not two's-complement nibbles.
For INT2, logical codes `[-2, -1, 0, 1, -1]` with `block_size = 16`
occupy one four-byte blob `[0xe4, 0xa9, 0xaa, 0xaa]` when padding is encoded
as logical zero. The four stored codes 0, 1, 2, and 3 decode to -2, -1, 0,
and 1; do not sign-extend the two-bit field as a two's-complement value.
All padding beyond logical K, including an odd-K INT4 high nibble or
unused INT2 lanes when K is not divisible by four, is ignored
regardless of its stored value. Reuse offline-quantizer padding bytes
unchanged; no padding-value host readback is needed for runtime B.

Decode into a wider integer before converting the centered code to INT8.
Do not quantize weights again, normalize block scales, or reconstruct
different INT8-domain weight values inside the operator. MatMulNBits
symmetric quantizers may emit negative scales, and all-zero blocks may
have zero scales. Preserve those values and the original scale dtype;
requiring positive scales or replacing zero scales would break reuse.

Use the same B shape helpers and scale-shape validation as the symmetric
subset of [MatMulNBits input validation](../../../onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_helper.h),
with checked arithmetic and this operator's explicit bit-width restriction.
The current helpers use unchecked `k + block_size - 1`, `block_size * bits`,
and `n * k_blocks` expressions. Guard those operations before calling them
or introduce checked equivalents; reusing the helpers alone does not satisfy
the size-safety requirement in section 2.3.
The existing [offline block quantizer](../../../onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py)
can supply standard `is_symmetric=True` QOperator weight/scales initializers
for bits 2, 4, or 8 without repacking or requantization.
This does not include asymmetric zero points, `g_idx`, or CUDA prepacked
layouts. An exporter may instead opt into the separately validated SM80
`weight_prepacked=1` contract for the supported SM80+ W2 block-64, W4 block-32,
or W8 block-32 geometry. The
`weight_prepacked=2`, QServe, GGUF, and Marlin formats are not implicitly
accepted. Internal prepacking must preserve the decoded logical codes and
scale values.
This is initializer compatibility, not equivalence to weight-only
MatMulNBits output: the new operator always quantizes A.

All three widths are part of the proposed version-1 contract, not reserved
formats. Further widths require a future schema extension defining their
storage, integer-dot backends, accumulator bounds, export, and tests;
do not retroactively reinterpret version-1 inputs.

## 3. Numerical Contract

All floating-point operations below use round-to-nearest-even and gradual
underflow, including FP32 scale products, interval additions, and bias
addition. Preserve subnormal inputs and results; a build's fast-math or
flush-to-zero settings must not change this contract. FP16/BF16 conversion
must likewise preserve representable subnormals.

Flatten A's leading dimensions into M independent rows. This matches the
existing [MatMulComputeHelper](../../../onnxruntime/core/providers/cpu/math/matmul_helper.h)
handling of `[..., K]` multiplied by a transposed two-dimensional weight.

Weight scales must be finite, including negative and zero values used by
symmetric MatMulNBits quantizers; report NaN/infinite scales as
`INVALID_ARGUMENT`, never replace them with a default. Section 5 specifies
when scale values are validated and the resulting graph-capture restriction.
Convert FP16/BF16 scales to FP32 exactly. A zero weight scale does not
suppress the specified NaN propagation from a nonfinite activation domain.

Finite activation values use the quantization rule below. If any activation
in a reduction domain is NaN or infinity, mark that domain's scale as a quiet
NaN and write zero activation codes for the entire domain. Its scaled
contributions then propagate NaN without any nonfinite float-to-integer
conversion. This is defined numerical propagation, not a successful substitute
for a rejected weight scale:

- Per-tensor mode produces NaN throughout the output.
- Per-token mode produces NaN in the affected row only.
- Per-block mode produces NaN in the affected row's output because that
  block contributes to every output column, even if its weight codes are zero.

Bias follows IEEE floating-point addition, including NaN/infinity propagation.
NaN payload bits are not part of the contract. Empty-output and empty-K cases
use the explicitly defined empty-reduction behavior.

### 3.1 Dynamic activation granularities

The granularity controls only the set of activation values sharing a scale.
All three modes use the same symmetric absmax, rounding, and integer range:

| Mode | Reduction domain for one scale | Internal scale shape | Batch independent? |
|------|--------------------------------|----------------------|--------------------|
| `"dynamic_per_tensor"` | All valid elements of A, across every row and K value. | Scalar | No; another row can change the shared scale. |
| `"dynamic_per_token"` | All K values in one flattened activation row. | `[M]` | Yes |
| `"dynamic_per_block"` | Consecutive K values within one row, in blocks of `activation_block_size`. | `[M, G_A]` | Yes |

For block mode, `G_A = ceil(K / activation_block_size)`. Partial final blocks
use only their valid values. An activation block is per token, not a group of
tokens. Weight and activation block sizes are independent; the generic path
supports unequal sizes and nonaligned boundaries.

Per-token mode is the default and matches issue #33200. Per-tensor mode can
reduce scale storage, but trades batch independence and potentially accuracy
for one shared scale; it is not the existing asymmetric UINT8
DynamicQuantizeLinear algorithm. Per-block mode localizes outliers within K
at the cost of more scales and partial-sum scaling work.

Dynamic per-channel activation quantization across M, UINT8/min-max variants,
and asymmetric variants are outside the initial contract. In particular,
asymmetric quantization needs zero-point correction and cannot be added as
another spelling of a symmetric granularity.
Offline SmoothQuant input-feature rescaling is not this excluded dynamic
per-input-channel mode. Neither a backend nor an exporter may rename
per-token activation quantization as per-channel.

### 3.2 Shared quantization rule

For a selected activation reduction domain D:

```text
amax[D] = max_{(m,k) in D} abs(float(A[m,k]))
sA[D]   = float32(amax[D] / 127)
qA[m,k] = clamp(round_to_nearest_even(float(A[m,k]) / sA[D(m,k)]), -127, 127)
```

`D(m,k)` selects the tensor, row, or row/K-block scale according to the
serialized mode. For an all-zero domain, define `sA[D] = 1` and all its
activation codes as zero. For a nonzero domain whose scale division
underflows to zero, use the smallest positive FP32 subnormal as its scale.
The reference uses gradual underflow; the quantization implementation must
not silently enable flush-to-zero.

Use a well-defined FP32 division and round-to-nearest-even conversion.
Approximate reciprocals are not interchangeable at quantization boundaries
unless their codes are proven equivalent. Reductions must exclude physical
padding and values outside the selected domain.

Adding, removing, reordering, or changing unrelated rows must not change a
row's quantization or output in per-token and per-block modes. In per-tensor
mode, adding or changing another row can intentionally change both.

### 3.3 Single scale interval: fused epilogue

When neither activation nor weight scales change along K for a given output,
the entire contraction is one integer dot product. This includes per-tensor
or per-token activations with per-channel weights, and block activations
when both activation and weight have just one K block.

Let `sA_for_row[m]` select the shared tensor scale or the row's single scale:

```text
acc[m,n] = sum_k int32(qA[m,k]) * int32(qB[n,k])
scaled[m,n] = ScaleProduct(acc[m,n], sA_for_row[m], float32(scales[n,0]))
Y[m,n] = cast_T(float32(scaled[m,n] + float(bias[n])))
```

Integer accumulation is exact within the supported accumulator range.
The shared scaling helper fixes the rounding order and avoids avoidable
overflow in the activation-scale intermediate:

```text
v = float32(acc)
t = float32(v * sA)
if acc == 0 or t is finite and normal:
    ScaleProduct(acc, sA, sB) = float32(t * sB)
else:
    ScaleProduct(acc, sA, sB) = float32((float64(v) * float64(sA)) * float64(sB))
```

For a nonfinite activation domain, return a quiet NaN instead of evaluating
this finite-domain rule. Check that marker before the zero-accumulator case.
For finite domains, the FP64 branch handles a nonzero accumulator whose first
product is infinite, subnormal, or zero. Products of the supported finite
FP32 operands and accumulator fit FP64's exponent range. The final FP32
conversion can still overflow or underflow when the scaled value itself is
out of range; output conversion to FP16/BF16 can likewise overflow.

For example, four maximum-finite BF16 activations, weight codes equal to 1,
and `scales = float32(1e-38)` have a finite result near 13.56. Multiplying
the INT32 accumulator by the activation scale first overflows FP32 and
would incorrectly produce infinity without the guarded wider branch.

Do not contract scaling and bias addition into an FMA. Conversion from a
large INT32 accumulator to FP32 may round and is part of the contract.
Absent bias means zero. `cast_T` uses round-to-nearest-even, including the
target type's subnormal and overflow behavior. All backends use this same
helper; the wider branch is not a dispatch-dependent numerical mode.

Both scales are constant across the entire dot product for a given output
element. They can therefore be applied after GEMM, in registers, without
materializing a global INT32 result.

### 3.4 General case: common scale intervals

If either operand's scale changes along K, form intervals from the sorted
union of activation and weight block boundaries, including 0 and K. For
per-tensor/per-token activation modes, only weight boundaries subdivide K.
Every resulting interval p has constant activation and weight scales.

```text
acc_p[m,n] = sum_{k in interval p} int32(qA[m,k]) * int32(qB[n,k])
term_p    = ScaleProduct(acc_p[m,n], sA[m,p], sB[n,p])
```

Here `sA[m,p]` and `sB[n,p]` select the original scales covering the interval,
with weight scales converted to FP32;
they do not require materializing expanded scale tensors. Combine `term_p`
in increasing K-interval order with FP32 additions, then add bias in FP32
and cast to T once. For a nonempty contraction, initialize the floating-point
total from the first interval term.

For example, with `K = 256`, activation block size 96, and weight block size
128, the interval boundaries are `[0, 96, 128, 192, 256]`. Equal block
indices cannot be used to pair the two scale tensors.

**A single final INT32 sum followed by one scaling epilogue is incorrect
when either operand's scale changes across K.** This includes per-channel
weights with multiple activation blocks. An MMA tile must not combine
products across a common scale boundary before scaling the partial sum.
For example, interval dots +2 and -2 have full-K sum zero, but weight
scales 1 and 2 yield -2; no final scalar can recover that result from zero.
This rule applies to INT8 and decoded INT4/INT2 weights.

Grouping is defined by the schema, not the CUDA tile shape. Do not merge
adjacent intervals merely because their scale values happen to be equal:
that can change the specified FP32 rounding of partial sums.

### 3.5 Accumulator limits and empty dimensions

With activation codes bounded by 127 and signed weight magnitudes bounded
by `2 ** (bits - 1)`,
an input-independent INT32 safety bound for one common scale interval is:

```text
S_A = activation_block_size for dynamic_per_block, otherwise K
L_bound = min(K, block_size, S_A)
B_magnitude = 2 ** (bits - 1)
L_bound * 127 * B_magnitude <= INT32_MAX
```

| Weight width | Maximum product magnitude | Maximum safe interval length |
|--------------|---------------------------|------------------------------|
| INT8 | `127 * 128 = 16256` | 132104 |
| INT4 | `127 * 8 = 1016` | 2113665 |
| INT2 | `127 * 2 = 254` | 8454660 |

Validate the bound using checked wider arithmetic. Version 1 reports
`INVALID_ARGUMENT` when this conservative bound is exceeded; do not wrap or
use saturating integer accumulation. It applies to each partial accumulator,
not the entire K when scales subdivide the contraction.
Supporting longer intervals with a wider exact accumulator
would require an explicitly specified extension.
For INT2, the valid weight-block limit already bounds every common
interval to 256 elements, well below its arithmetic limit. Large total K
remains supported through multiple independently scaled intervals.

Validate types, shapes, attributes, and checked sizes before either empty
case. An empty output returns without launching compute. For `K == 0`, the empty
sum is zero and Y is zero plus optional bias; no row quantization or GEMM is
needed. With MatMulNBits storage, B has shape `[N, 0, blob_size]` and scales
have shape `[N, 0]` or `[0]` in this case; `block_size` remains valid.
Runtime scale-value validation is skipped for these cases, as shown in
section 4; constant scales are still validated during session initialization.

## 4. CUDA Dispatch

Native dispatch requires runtime compute capability at least 80 and loaded
kernel binary/PTX versions at least 80. All native kernels use signed INT8
`mma.sync.aligned.m16n8k32`, with fused scaling, bias, and output conversion:

- Single-interval W8A8: `64 x 128 x 64` CTA tile, two-stage `cp.async` staging,
  `ldmatrix` fragment loads, register XOR decoding, and bank-distributed shared storage. Requires
  M at least 16, N at least 8, K at least 32 and divisible by 16, K fitting
  a bounded `int` loop counter, and both block sizes at least K. SM90+ uses
  a K tile of 128 when M and K are at least 128. Dynamic shared storage is
  30,720 or 55,296 bytes per CTA, respectively.
- Aligned grouped W8A8/W4A8/W2A8: `32 x 64 x 64` or `64 x 64 x 64` CTA tile,
  with two-stage asynchronous copies of still-packed weights. Four codes
  are decoded together in registers. Requires M at least 32, N at least 8,
  K and weight blocks divisible by 64, and activation blocks at least K or
  divisible by 64. Loop dimensions must fit bounded `int` counters; address
  products remain 64-bit. The larger M tile is selected when its grid has
  at least 256 CTAs. Scales are reused across fragments, and INT32 dots are
  reset at each common scale boundary before ordered FP32 accumulation.
  SM80 (A100) instead accepts any M, chooses among `16 x 32`, `16 x 64`,
  `32 x 64` and `64 x 64` tiles by row count and CTA count, and stages
  128-code K tiles when K and the weight/activation blocks are multiples of
  128 (section 8.16).
- Other native W8A8/W4A8/W2A8: `32 x 64 x 64` CTA tile with direct shared-memory
  weight decoding. Requires M at least 32, N at least 8, K at least 32,
  and every interior activation/weight block boundary 32-aligned: each
  block size must be at least K or divisible by 32.

M/N tails and final partial K tiles are predicated. Unsupported geometry
uses the one-warp-per-output DP4A/scalar path; all paths accumulate common
intervals in the same ordered FP32 sequence.

The single-interval M threshold was checked against DP4A on the measured
H200 shapes, but is not universally tuned. Aligned pipeline tile choices
were measured on H200; SM80 (A100) and SM120 have their own measured choices
(sections 8.15 and 8.16), and other architectures remain unverified.
The remaining native geometry uses synchronous shared staging.

Dispatch selects an implementation, never the activation-quantization mode.
Using the new operator is the model's W8A8/W4A8/W2A8 opt-in; the serialized attribute
selects its dynamic granularity. No mode disables activation quantization,
and no environment variable is needed to enable it.

```text
validate types, shapes, attributes, dimensions, and accumulator limits
    |
    +-- empty output ------------------------> return
    +-- K == 0 ------------------------------> zero plus optional bias
    |
    +-- ensure weight scales are validated under the capture policy
    |
    +-- quantize A using the selected tensor/token/block domains
            |
            +-- tuned small-M case ----------> integer GEMV/small GEMM
            +-- one interval, native ready --> integer tensor-core GEMM + epilogue
            +-- multiple intervals, ready ---> interval-partial integer GEMM
            +-- otherwise ------------------> general integer CUDA fallback
```

Fast-path eligibility includes the compiled kernel's availability, device
capability, activation/output dtype, weight width and packing, alignment,
dimensions, and scale layout. A device-major
comparison alone is not sufficient evidence that a native kernel is available.

Unsupported fast-path geometry routes to the generic implementation.
An actual launch, allocation, or CUDA-library error is returned to ORT;
do not hide an execution failure by retrying another backend.

### 4.1 Activation quantization

Share the quantization rule and conversion helpers across native, small-M,
and fallback paths. Produce:

- `qA`: `[M, K_padded]` INT8, with zero-filled internal K padding.
- `sA`: scalar, `[M]`, or `[M, G_A]` FP32, according to the selected mode.

The current implementation stores contiguous `[tile_M, K]` activations
without global padding. Native K-tail zero fill is confined to shared
memory, and unaligned row strides use predicated byte loads.

For per-token mode and ordinary transformer K sizes, a CTA per row with
vectorized loads is a reasonable initial implementation. For per-block
mode, a CTA or warp group can handle each row/K-block domain. Multi-CTA
reductions for larger domains must keep the same quantization reference.

Per-tensor mode requires a reduction over the complete valid A tensor,
followed by quantization using its one final scale. Use device-resident
partial maxima and final reduction; do not derive a separate scale per CTA
or per M tile, and do not synchronize the scale back to the host.

Do not repeat activation reduction separately for every output-column tile.
The initial pipeline quantizes before GEMM inside the ORT operator; it does
not claim to fuse reductions into GEMM. Per-tensor reduction may need more
launches than the per-token quantize-plus-GEMM pipeline.

### 4.2 Primary per-channel GEMM

The contributed Ada-oriented kernel is a suitable initial backend:

- Signed INT8 tensor-core multiplication with INT32 accumulation.
- Decode B's unsigned offset codes to centered INT8 fragments for all three
  widths; unpack INT4 nibbles or INT2 two-bit fields before the same
  INT8 x INT8 MMA. Do not assume a mixed INT8-activation/INT4- or
  INT2-weight instruction exists on the target device.
- `mma.sync.m16n8k32`, `ldmatrix`, and a `cp.async` pipeline where supported.
- A starting CTA tile of `128 x 128 x 64` with three pipeline stages.
- Predicated loads/stores for M/N tails and zero-filled internal K tails.
- The tensor-wide or per-row activation scale and column weight scales
  loaded for each output fragment.
- The shared scaling helper, normally FP32 with a guarded FP64 branch for
  extreme intermediates, followed by FP32 bias addition and direct
  FP16/BF16 output stores.

Tile sizes and pipeline depth are backend details, not schema attributes.
Benchmark before establishing crossover thresholds or enabling a tile on
another architecture. CUTLASS and cuBLASLt backends can be added later under
the same numerical contract; the contributed backend need not introduce
a new dependency.
Other integer instructions may be used only when they preserve the same
decoded-code dot products and accumulator contract.
INT2 single-interval eligibility additionally requires a valid single
weight block, so typical transformer K sizes use the multi-interval path.

### 4.3 Small-M and block-scaled kernels

For decode and short speculative batches, use integer GEMV or a smaller
GEMM when measured latency is lower. Reuse qA and sA. A DP4A/scalar integer
path is a valid baseline; choosing small M must not bypass activation
quantization or change its granularity.
Decode the MatMulNBits offset codes before DP4A or scalar multiplication.

For multiple scale intervals, reset integer accumulation whenever either
operand's scale changes and retain a running FP32 output accumulator.
Support arbitrary positive activation block sizes and all MatMulNBits-valid
weight block sizes in the generic path, even if native tiles require
sizes aligned to their MMA K extent or
matching block boundaries. Per-channel weights alone do not make
multi-block activations eligible for a final-epilogue-only kernel.
INT2 blocks 16/32 remain schema-valid even if a native kernel needs a
32/64-element MMA K tile. Such kernels must isolate scaled partials at
every common boundary or route to the generic integer path. Existing
MatMulNBits weight-only kernel eligibility is not evidence that a
W2A8 backend supports the same geometry.

Do not reorder FP32 block reductions or use nondeterministic atomic
split-K accumulation in a backend advertised as bit-exact.

### 4.4 Correctness fallback

W8A8/W4A8/W2A8 fallback choices are:

1. A direct CUDA integer-dot kernel with the specified scaling order.
2. INT8 GEMM into a bounded INT32 tile, followed by a scale/bias kernel.
3. For multiple intervals, integer partial GEMMs with FP32 interval accumulation
   and a final bias/conversion step.

The existing
[GemmInt8 helper](../../../onnxruntime/core/providers/cuda/integer_gemm.cc)
is potential reusable infrastructure. It currently consumes non-transposed
logical B in `[K, N]` layout, so it cannot directly consume this operator's
logical `[N, K]` weights. Reuse requires a transpose-aware extension or
explicit internal repacking and validation of CUDA-library restrictions.
For all three widths, decode the UINT8 offset representation into bounded INT8
weight tiles when the fallback GEMM requires them. An INT8 helper must
never reinterpret stored UINT8 B as a signed INT8 matrix.

**Dequantize to FP16/BF16 followed by floating GEMM is not an exact W8A8/W4A8/W2A8
fallback.** It rounds intermediate values and changes accumulation semantics.
It is not a backend of this dynamic-only operator.

## 5. Memory, Prepacking, and Streams

The implementation uses a 16 MiB target for quantized activations,
their scales, and tensor-reduction scratch, or one complete row when that
row exceeds the target. It
reuses each row tile across output columns, decodes B directly without
expanded-weight scratch, and keeps integer/FP32 output accumulators in
registers. Per-tensor absmax is computed once over all of A before tiling.
Per-tensor absmax uses a bounded multi-CTA reduction with at most 1,024
partial maxima (4 KiB), followed by a device reduction to the global scale.
The native path decodes packed weights into shared memory per tile; no
expanded global weight matrix or global INT32 output is allocated.
Backend-specific weight prepacking is not implemented.

Use ORT's stream-aware scratch allocation and the context's compute stream.
The single-interval native path requires quantized activation storage and
activation scales, but no `[M, N]` INT32 scratch output.

Generic paths must tile M and/or N to bound INT32 and FP32 scratch, rather
than allocating unbounded output partials for every scale interval. Follow
the FP8 operator's bounded-scratch approach, but keep integer accumulation.
When M is tiled, quantize only the active rows and reuse them across N tiles.
Per-tensor mode must compute its global scale before any such tiling; tiling
may change storage and launch geometry, never the quantization domain.

Activation-scale storage is one float for per-tensor, M floats for per-token,
and `M * G_A` floats for per-block. Include scales and reduction scratch in
the memory budget, and retain only the active row tile's block scales when
possible.
Include offset-code conversion and INT4/INT2 unpacking/repacking scratch in
the budget. The native path should decode into registers/shared memory rather than unconditionally
materializing a full expanded `[N, K]` INT8 weight matrix. A generic path
may use bounded expanded tiles while preserving the same logical codes.

Use `PrePack` only for constant weights/scales that need backend-specific
padding, transposition, or reordering. Such formats remain internal:

- Do not serialize an Ada-specific or Blackwell-specific layout in B.
- Preserve a usable representation for fallback backends and all three widths.
- Runtime weights/scales must not reuse stale initializer-derived data.
- Shared prepacking, if implemented, must identify dtype, dimensions, block
  size, weight bit width, packing format/version, and relevant architecture
  constraints.
- Initialization work must complete before packed buffers are consumed on
  an inference stream; do not add inference-time device-wide synchronization.

Keep per-run activation scratch local to the invocation. Avoid mutable
kernel-owned activation buffers that race across concurrent runs.
Activation reductions, nonfinite-domain marking, scale generation, and
GEMM remain device-resident. Do not introduce a per-run device-to-host
synchronization for activation values or dynamically generated scales.

### 5.1 Weight-scale validation and graph capture

Value validation and synchronous ORT error reporting require a host-visible
result. The initial implementation makes that cost and capture limit explicit:

- Validate non-overridable constant `scales` initializers once during session
  initialization. If device-resident, a validation kernel and a checked
  device-to-host flag transfer may synchronize the initialization stream.
  Preserve the validated initializer or its packed representation.
- For runtime or overridable `scales`, validate every invocation with
  nonempty computation. A checked validation flag transfer and compute-stream
  synchronization are allowed outside capture, and must be included in
  end-to-end benchmarks for runtime scales. Do not use device-wide synchronization.
- CUDA graph capture requires an already validated, non-overridable constant
  `scales`. Reject runtime/overridable scales during capture before issuing
  a host readback or synchronization. Warmup validation of a mutable pointer
  is not enough, since replay could supply different values at that address.
- Runtime A and B remain compatible with capture under that constant-scale
  restriction. Nonfinite A is handled by the captured device kernels and
  the specified NaN propagation rule, not a host-side error check.

Capture/replay must also follow ORT's stable device-buffer binding requirements.
Shapes and bound addresses of A, B, and Y, and the captured scratch addresses,
remain fixed across replay. Changing values in those buffers is supported;
changing shapes or substituting newly allocated buffers requires recapture.

Do not cache validation based only on pointer identity or assume a
user-supplied runtime scale tensor stays unchanged. This restriction is part
of the proposed capture support, not an implicit success-shaped fallback.

## 6. Compatibility and Model Integration

Issue #33200's option 2 is preferred over changing existing operators:

- [DynamicQuantizeMatMul](../../../onnxruntime/core/graph/contrib_ops/quantization_defs.cc)
  has a float activation contract, conventional MatMul weight layout, and
  an existing per-tensor UINT8 activation path. A new attribute alone would
  not cover the proposed dtype, layout, and quantization changes.
- `MatMulIntegerToFloat`, defined in the same schema file, does not currently
  specify the required per-row activation-scale contract. Adding such a
  mode would need separate schema and implementation work.
- [MatMulNBits](./matmul_nbits.md) has its own packed-weight, zero-point, and
  weight-only contract. Keep W8A16/W4A16/W2A16 there rather than duplicating it in the
  new operator. Its existing `accuracy_level` attribute permits internal
  activation quantization on supporting backends, but does not specify this
  proposal's granularity and rounding contract. This proposal must not change
  that behavior or add implicit W8A8/W4A8/W2A8 dispatch to existing nodes.
- Existing FP4/FP8 nodes retain their current semantics and input positions.
- Reuse standard symmetric MatMulNBits B/scales initializers for all three
  widths. MatMulNBits accepts floating-point A; UINT8 describes its packed
  weights. Initializer reuse does not carry over optional asymmetric zero
  points or its existing activation-computation semantics.

The default offline quantizer supports explicit per-node opt-in for floating
MatMul nodes with known FP16/BF16 activation types:

```python
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

config = DefaultWeightOnlyQuantConfig(
    bits=8,
    block_size=8192,
    is_symmetric=True,
    activation_quantization="dynamic_per_token",
)
quantizer = MatMulNBitsQuantizer(
    model,
    nodes_to_include=["projection"],
    algo_config=config,
)
quantizer.process()
```

`activation_quantization=None` (the default) retains weight-only MatMulNBits
export. The opt-in requires symmetric QOperator quantization, the default
algorithm, and MatMul-only selection; it rejects incompatible types, invalid
block sizes, and source-only `accuracy_level` settings. For block mode,
also set a positive `activation_block_size`. Packed weights and scales are
byte-identical to the corresponding symmetric weight-only export.
The CLI exposes the same `--activation_quantization` and
`--activation_block_size` options. This exports new MatMul nodes; conversion
of already quantized MatMulNBits nodes is not implemented.

A manually constructed node uses the same contract:

```python
node = onnx.helper.make_node(
    "DynamicQuantizeMatMulNBits",
    ["A", "B", "scales", "bias"],
    ["Y"],
    domain="com.microsoft",
    bits=8,
    block_size=8192,
    activation_quantization="dynamic_per_token",
)
```

For logical K = 6144, B is UINT8 `[N, 1, 8192]` and scales are `[N, 1]`
or `[N]`. The remaining 2048 stored codes per row are ignored padding. Omit the
fourth input when bias is absent. The model must import version 1 of the
`com.microsoft` domain once the proposed operator is implemented.

For INT4 block weights, reuse the existing symmetric MatMulNBits
initializers and select the actual weight width explicitly:

```python
node = onnx.helper.make_node(
    "DynamicQuantizeMatMulNBits",
    ["B_input", "B_Q4", "B_scales"],
    ["Y"],
    domain="com.microsoft",
    bits=4,
    block_size=128,
    activation_quantization="dynamic_per_token",
)
```

Here `B_input` is the floating activation, `B_Q4` has shape
`[N, ceil(K / 128), 64]` and type UINT8, and `B_scales` retains its original
FP16/BF16/FP32 values and `[N, G]` or `[N * G]` shape. Unlike the
per-channel example, this case needs weight-block partial-sum scaling
even though A is quantized per token.

For INT2, use the same export pattern with `bits=2`, a symmetric `B_Q2`
initializer, and a valid weight block size. With `block_size=128`, B has
shape `[N, ceil(K / 128), 32]`, four stored codes per byte, and implicit
midpoint 2. Reuse its scales unchanged. The default activation mode is still
per-token INT8; INT2 changes only the weight width and decoding.
Weight blocks larger than 256 are invalid, including attempts to encode
per-channel INT2 weights for K greater than 256.

To use a single dynamic scale for all A values, select
`activation_quantization="dynamic_per_tensor"` instead. For per-token K-block
activation quantization, select:

```text
activation_quantization = "dynamic_per_block"
activation_block_size = 128
```

The activation block size is independent of `block_size=8192`: this example
has per-channel weights and 48 activation scales per token. It needs
scaled interval partial sums, not the per-channel final-epilogue-only kernel.

Do not fuse a floating MatMul or the existing UINT8 dynamic-quantization
pattern into this node unless the graph already explicitly selects the
same symmetric granularity, ranges, and rounding semantics. Node-level selection is necessary
because activation outliers and smoothing requirements are model dependent.

Conversion from a symmetric MatMulNBits node must copy its actual `bits`
and `block_size`, not rely on defaults: MatMulNBits defaults to 4 bits,
whereas this proposal defaults to 8. Reuse B/scales tensors and remap bias
from MatMulNBits input 5 to input 3. Source zero points must be absent or
non-overridable constants verified to equal the fixed symmetric midpoint
before omitting that input. Decode packed zero points and check only valid
block entries; unused padding lanes need not equal the midpoint. Reject
runtime or overridable zero points, even if their current values match.
Reject nondefault zero points, `g_idx`, and prepacked layouts other than the
explicitly supported `weight_prepacked=1` W2 block-64, W4 block-32, and W8 block-32 geometries rather than
silently dropping their meaning. Check that source A/output and optional bias
use the supported FP16/BF16 type, and that source `K`/`N` attributes agree
with A and the canonical B/scales shapes before removing those attributes.
An FP32 MatMulNBits node cannot be converted by renaming it; changing its
activation/output dtype requires a separately selected model transformation.
Do not copy `accuracy_level` or other source-only attributes into the new
schema. This conversion is explicit W8A8/W4A8/W2A8
opt-in, not a semantics-preserving rewrite of a weight-only node.

### 6.1 Fusion scope and priorities

Version 1 exposes only optional bias; additional entries below require a
separate schema/optimizer extension and are not implicit behavior of this
operator. Apply output operations after all scaled K-interval contributions.

| Priority | Fusion | Scope and constraint |
|----------|--------|----------------------|
| P0, version 1 | GEMM scaling + bias + output conversion | Scale interval partials where needed, then add bias in FP32 and cast once; no global INT32 output on the native single-interval path. |
| P1, follow-up | GEMM + residual Add; GEMM + pointwise activation | Elementwise epilogues once the scaled sum is final. Define residual broadcasting, activation variants, arithmetic order, and rounding before adding inputs/attributes. |
| P1, follow-up | Residual + RMSNorm + dynamic quantization | Producer-side fusion; share normalized quantized activations across projections rather than repeat normalization/quantization inside each GEMM. |
| P1, follow-up | SiLU-and-multiply + dynamic K-block quantization | FFN intermediate producer fusion; compute scales over the same selected block domains. |
| P2, specialized | Gate/up projections + SwiGLU/GEGLU | Requires paired projection outputs/layout and measured shape-dependent dispatch; do not treat gating as an arbitrary unary activation. |
| P2, specialized | GEMM output + dynamic requantization | Requires explicit quantized outputs/scales and completion of the scale domain; do not hide it behind floating Y. |
| Deferred | GEMM + RMSNorm/LayerNorm | Whole-output-row reduction is not a normal tile-local GEMM epilogue. |

Pointwise epilogues need no additional row reduction. Dynamic output
per-token quantization, however, must see every N value in the output row;
per-block quantization must see its entire output block. Multiple CTAs
owning one domain need an explicit reduction strategy or a specialized
ownership scheme. Merely adding an epilogue attribute does not solve this.

Preserve materialization/rounding semantics when replacing an existing
graph: casting a GEMM output before bias, activation, or quantization can
differ from keeping the intermediate in FP32. A future fused contract
must specify which values are reduced and when conversion occurs.
Kernel availability in another repository does not establish automatic
compiler fusion or performance across all devices and shapes.

### 6.2 Quantization reuse

If multiple GEMMs share A, independently quantizing A is valid but may repeat
work. Plan for optimizer-internal reuse or a separate reusable
dynamic-quantization operator and externally quantized GEMM. Those interfaces
are follow-up work, not prerequisites for version 1, and must not make optional
`a_scale` presence change this dynamic operator's meaning.
Do not share a quantized activation or scale across nodes with different
granularities, activation block sizes, rounding rules, or producer semantics.

## 7. Implementation Surfaces

The implementation should follow the neighboring FP4/FP8 operator structure:

| Surface | Required work |
|---------|---------------|
| [Contrib schemas](../../../onnxruntime/core/graph/contrib_ops/contrib_defs.cc) and [Microsoft opset registration](../../../onnxruntime/core/graph/contrib_ops/ms_opset.h) | Add and register schema documentation, UINT8 B and floating scale type constraints, bit-width/shape validation, attributes, and shape inference. |
| [CUDA math operators](../../../onnxruntime/contrib_ops/cuda/math/) | Add `dynamic_quantize_matmul_nbits.cc` and `.h` for the operator and dispatch, plus a new `.cu` file for integer backends with all three MatMulNBits offset-code decoders. |
| [MatMulNBits layout helpers](../../../onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_helper.h) | Reuse canonical B/scale layout rules with new checked size calculations or guards around the existing unchecked helpers, without importing asymmetric/group-index execution semantics. |
| [CUDA contrib registration](../../../onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc) | Add declaration and kernel-create registration. |
| [Standard CUDA build](../../../cmake/onnxruntime_providers_cuda.cmake) and [plugin CUDA build](../../../cmake/onnxruntime_providers_cuda_plugin.cmake) | Include the same operator and correctly gated native kernels in both builds. |
| [Shared CUDA source filtering](../../../cmake/onnxruntime_cuda_source_filters.cmake) | Isolate architecture-specific sources where needed; retain generic kernels when native sources are excluded. |
| [Quantization tooling](../../../onnxruntime/python/tools/quantization/) | Add explicit per-node export selection; reuse symmetric MatMulNBits B/scales initializers, copy bits/block size, and validate omitted source inputs. |
| [Contrib operator tests](../../../onnxruntime/test/contrib_ops/) | Add focused OpTester cases, offline-quantizer compatibility, all three weight-width decoders, and direct quantization/dispatch tests. |
| [Contrib benchmark harness](../../../onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py) | Extend or reuse the existing accuracy/latency harness for W8A8, W4A8, and W2A8, including decoding/padding costs. |
| [Contrib operator documentation](../../../docs/ContribOperators.md) | Regenerate the public operator entry after schema implementation. |

Do not make the new operator depend on FP4/FP8 type support or optional MoE
build flags. Windows/MSVC, Linux, and plugin builds need the generic operator
even when a particular native backend is not compiled.

## 8. Validation Plan

### 8.1 Numerical and shape coverage

- FP16/BF16; rank-1, rank-2, and higher-rank A; optional bias.
- All three weight widths and all three dynamic modes combined with
  per-channel (INT2 K <= 256) and block-scaled weights; all supported scale
  types and 1D/2D scale shapes.
- Unequal activation/weight block sizes, misaligned boundaries, and
  deliberately unequal scales on consecutive intervals.
- Stored INT8 codes 0/128/255 decoding to -128/0/127; all sixteen INT4
  codes, including 0/8/15 decoding to -8/0/7; negative activations and outputs.
- All four INT2 codes 0/1/2/3 decoding to -2/-1/0/1, all four bit lanes,
  and K modulo four equal to 0/1/2/3.
- INT4 low/high nibble order, odd K, INT2 low-two-bit-first order,
  packing restarted at each block/row, partial final blobs, and arbitrary
  padding bytes that must not affect Y.
- Symmetric MatMulNBits offline-quantizer initializers reused byte-for-byte
  for all three widths, including negative scales and zero-scale blocks.
- All-zero rows, very small finite rows, large finite values, and RNE ties.
- Extreme BF16/scaling combinations that overflow the first FP32 product
  despite a finite final result, plus the guarded wider branch for subnormals.
- Subnormal weight scales with a normal first scale product, subnormal
  FP32 interval totals, and representable FP16/BF16 subnormal outputs;
  verify gradual underflow on both scaling branches and during conversion.
- Quantization values immediately on both sides of a rounding boundary.
- M/N/K tails and non-power-of-two K; valid power-of-two weight blocks;
  partial final blocks and arbitrary positive activation block sizes.
- Empty M/N, `K == 0`, and accumulator-bound acceptance/rejection.
- INT8 interval lengths 132104/132105 and INT4 lengths 2113665/2113666
  accepted/rejected by the checked conservative bound, respectively.
- INT2 arithmetic-bound helper coverage at 8454660/8454661; schema-valid
  INT2 intervals cannot reach this limit because weight blocks are at most 256.
- INT2 weight blocks 16/32/64/128/256 accepted, 512 rejected, and large K
  handled by multiple intervals without collapsing weight scales.
- Invalid ranks, dimensions, B types/blob shapes, scale types/shapes,
  non-power-of-two or sub-16 weight block sizes, attributes, and bias types/shapes.
- Explicit errors for NaN or infinite weight scales, including an invalid
  runtime scale supplied after a valid invocation; finite negative and zero
  scales must be accepted unchanged.
- NaN/infinite activations in each granularity, with the specified affected
  rows and no dependency on NaN payload bits; nonfinite bias propagation.
- Default and explicit `bits = 8` equivalence; explicit `bits = 4` and
  `bits = 2` decoding; rejection of unsupported widths such as 1/3/16.
- Export rejection of asymmetric zero points, group indices, and CUDA
  prepacked source weights; explicit fixed-midpoint input omission.
- Export rejection of runtime/overridable midpoint zero points, FP32
  activations, and inconsistent source K/N; accept arbitrary unused packed
  zero-point padding and remove source-only attributes.
- Rejection of the removed `"none"` mode and missing/extra activation block
  size attributes.
- Batch independence for per-token/per-block, including an extreme outlier
  added in another row; intentional shared-scale changes for per-tensor.
- Per-tensor parity when M is scratch-tiled, proving its scale is still global.
- Per-block locality: changing another activation block must not change the
  unchanged block's scale or quantized values.

Build the W8A8/W4A8/W2A8 reference from the actual FP16/BF16 input values, not
their pre-conversion FP32 sources. Decode the canonical MatMulNBits blobs,
use the actual stored scale values converted to FP32, compute integer dot
products in a wider host type, check the supported bound, and apply the
specified FP32 ordering.
Test qA codes exactly; output closeness alone can miss quantization mistakes.

For forced W8A8/W4A8/W2A8 backends, require identical activation codes, decoded
weight codes, integer partial sums, and non-NaN output bits against the
reference. Compare NaN classification, not payload bits, including when
finite inputs produce opposite-signed overflowing interval terms whose sum
is NaN. Also check the specified propagation from nonfinite activation domains.
Exercise both branches of `ScaleProduct` and FP16/BF16 output rounding.
Use the common-interval reference, not an assumption that activation and
weight block indices are interchangeable.

### 8.2 Dispatch and execution coverage

- Force native, small-M, blocked, and generic paths through internal test
  controls for all three widths without adding public model attributes.
- Prove the intended native path executed; numerical agreement alone is not
  evidence of tensor-core dispatch.
- Test absent native build support, unsupported fast-path alignment, and
  generic-path routing without falling back to CPU or changing activation mode.
- Exercise constant and runtime weights/scales, prepacking, concurrent runs,
  and multiple sessions.
- Exercise CUDA graph capture/replay with validated constant scales and
  changing A/B values in stable, fixed-shape bound buffers; reject runtime
  or overridable scales before any prohibited
  capture-time synchronization. A prior warmup must not bypass that rejection.
- Build both standard and plugin CUDA providers; include Windows/MSVC coverage.
- Use compute-sanitizer for padded/tail cases and scratch lifetime checks.

### 8.3 Performance and model accuracy

Start with the issue's prefill cases, `M = 1536` and `(K, N)` equal to:

```text
(2048, 6144), (6144, 2048), (2048, 4096), (2048, 2048)
```

Also sweep small M, crossover sizes, ragged shapes, large N, and multi-block
weights for all three widths across all three activation granularities. Prioritize
the per-token baseline, then matched block sizes 32/64/128 for native W4A8
and multi-interval tuning, while retaining generic unequal-boundary coverage.
Benchmark W2A8 at every valid weight block size, especially 16/32 generic
fallbacks and 64/128/256 tuned paths; do not assume four codes per byte
guarantees a speedup after decoding and interval scaling. Compare model
accuracy separately for each weight width using the selected offline quantizer.
Validate on Ada and the
reported SM121 workload, with representative
Ampere/Hopper/Blackwell coverage where available. Do not assume identical
kernel eligibility or crossover points across those architectures.

Report reduction/quantization, weight decoding/unpacking, GEMM, fallback
scaling, and total operator latency.
Identify constant versus runtime scales and include runtime-scale validation
overhead. Measure the guarded wider epilogue separately on extreme-value cases;
it must not force the ordinary-value workload onto an FP64 GEMM.
Measure cuBLASLt INT8 plus its required scaling/output conversion, FP16 GEMM,
and MatMulNBits with clearly identified quantization settings. A comparison
against bare INT8-to-INT32 GEMM alone is not an end-to-end comparison.
For each weight width, compare the dynamic operator with MatMulNBits using
identical symmetric B/scales initializers. Report packed B bytes, per-channel
block-padding overhead, unpacking scratch, and prepack memory separately.
Do not attribute a weight-format change to activation quantization speedup.

Include warmup, repeated CUDA-event measurements, ORT session latency,
workspace usage, GPU/toolkit versions, and actual dispatch. Issue-reported
speedups are motivation, not verified performance guarantees for this proposal.
Model-level accuracy must be checked with the chosen per-node configuration,
including an outlier-sensitive case with and without offline smoothing.

### 8.4 Baseline implementation benchmark (2026-10-10 UTC)

Measured on an idle NVIDIA H200 (SM90), driver 580.105.08, CUDA 13.0.48,
cuDNN 9.19, PyTorch 2.13.0+cu130, and an ORT 1.31.0 Release source build
with SM86/SM90a kernels. These results describe the synchronous baseline;
the pipelined follow-up is measured in section 8.5.

The reproducible
[benchmark script](../../../onnxruntime/test/python/transformers/benchmark_dynamic_quantize_matmul_nbits.py)
uses device I/O binding and the same user compute stream as CUDA events.
Event and host measurements include launch/host gaps, but exclude
session initialization, transfers, and baseline weight preparation.
Prefill results use 10 warmups, 50 repetitions, five trials, and the
median per-call latency. B/scales are constant, bias is absent, and A/Y
are FP16. Weight blocks are 2048, except K=6144 uses 8192.

| M | K | N | Initial native operator (us) | Tuned operator (us) | FP16 GEMM (us) | Eager INT8 pipeline (us) |
|---|---|---|-----------------------------|---------------------|----------------|-------------------------|
| 1536 | 2048 | 6144 | 1402.15 | 279.70 | 55.42 | 535.45 |
| 1536 | 6144 | 2048 | 1526.03 | 357.61 | 54.24 | 654.32 |
| 1536 | 2048 | 4096 | 938.16 | 171.92 | 37.17 | 355.67 |
| 1536 | 2048 | 2048 | 484.12 | 120.33 | 19.79 | 254.00 |

The tuning improves these end-to-end cases by 4.0-5.5x and is 1.8-2.1x
faster than the unfused PyTorch INT8 pipeline, but remains substantially
slower than FP16 GEMM. The INT8 comparator includes absmax, quantization,
INT8 GEMM into INT32, scale application, and output conversion; it is
not bare INT8 GEMM. PyTorch's eager division can generate different codes
at rounding boundaries, so it is a performance comparator, not the exact
numerical oracle. The benchmark independently checks the operator against
FP32 NumPy division/rounding followed by an exact INT8 GEMM and the specified
scaling order for eligible single-interval cases. A separately tuned
cuBLASLt end-to-end implementation has not been measured.

Nsight Systems confirms `NativeSingleIntervalMatMul<half, 64>` executed.
For M/K/N=1536/2048/6144, traced activation quantization averaged 12.67 us
and native GEMM including decoding/scaling/output averaged 258.68 us.
Decoding and epilogue are fused, so there is no independent unpacking or
fallback-scale kernel. The non-profiled host median was 279.81 us.

The existing MatMulNBits baseline raises SIGFPE for the tested
M/K/N=1/2048/6144, block=2048 geometry in this checkout. The prefill sweep
explicitly uses `--skip-matmul-nbits`; no substitute weight representation
or weight-only speedup is claimed. Separate byte-identical initializer
comparisons below use the working block=128 geometry.

For M/K/N=1536/1024/1024, block=128, five warmups, 20 repetitions, and
three trials:

| Weight bits | Per-token operator (us) | Per-block A128 operator (us) | Per-tensor operator (us) | Weight-only MatMulNBits (us) |
|-------------|-------------------------|-----------------------------|--------------------------|-----------------------------|
| 2 | 224.28 | 245.15 | 230.29 | 24.68 |
| 4 | 236.80 | 254.44 | 240.29 | 24.61 |
| 8 | 214.94 | 231.42 | 218.84 | 26.08 |

These general native paths remain untuned and slower than weight-only
MatMulNBits. MatMulNBits uses the identical symmetric packed bytes and
scales with `accuracy_level=0`; it does not have equivalent dynamic
activation semantics. Additional sweeps cover M=1/16/128, BF16,
nonaligned activation blocks, ragged shapes, and runtime-scale validation.
Runtime-scale measurements include the validation kernel, readback, and
stream synchronization. For W8, block=128, K=N=1024, runtime scales increased
M=1 latency from 25.09 to 39.28 us and M=128 from 105.40 to 121.08 us.

Packed W8 bytes for the four prefill cases are 12/16/8/4 MiB respectively;
the K=6144 case includes 4 MiB of block padding. The corresponding FP16
baseline weight matrices use 24/24/16/8 MiB. Quantized activations and scales
use the 16 MiB row-tile budget; tensor reduction uses at most 4 KiB.
Expanded-weight scratch and prepack memory are zero. The single-interval
kernel uses 15,360 bytes of shared memory per CTA and no global INT32
output. The PyTorch INT8 baseline materializes expanded INT8 weights and
an M-by-N INT32 output, reported separately by the script.

Example reproduction, from outside the source package directory:

```bash
ORT_ROOT=/home/tianlei/git/onnxruntime
source "$ORT_ROOT/.venv_cu130/bin/activate"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$ORT_ROOT/build/gemm_autotune/Release"
export LD_LIBRARY_PATH="/home/tianlei/cuda13.0/lib64:/home/tianlei/cudnn9.19_cuda13/lib:$PYTHONPATH"
cd /tmp
python "$ORT_ROOT/onnxruntime/test/python/transformers/benchmark_dynamic_quantize_matmul_nbits.py" \
    --m 1 16 128 1536 --skip-matmul-nbits --warmup 10 --repeat 50 --trials 5 \
    --output benchmark_prefill_w8.json
python "$ORT_ROOT/onnxruntime/test/python/transformers/benchmark_dynamic_quantize_matmul_nbits.py" \
    --m 1 128 1536 --k 1024 --n 1024 --bits 4 --block-size 128 \
    --mode dynamic_per_block --activation-block-size 128
```

Validation on this machine: standard and plugin CUDA builds passed all
14 focused C++ tests each; nine Python methods plus 32 subtests passed,
including capture/replay, concurrency, initializer compatibility, and
export validation. Targeted integrated Compute Sanitizer checks reported
zero memory errors and zero race hazards. Standalone SM75/80/90 compilation
and 1,406 CPU/generic/native parity cases also passed.
Windows, Ada/Blackwell execution and model-level accuracy were not verified.

### 8.5 Pipelined implementation benchmark

The same H200, source build, benchmark script, input generation, stream,
warmup, repeat and trial settings as section 8.4 were used. Baselines were
remeasured before modifying the kernels. These are complete operator
CUDA-event medians, including activation quantization and host/launch gaps.

| M | K | N | Remeasured baseline (us) | Pipelined operator (us) | Speedup |
|---|---|---|-------------------------|------------------------|---------|
| 1536 | 2048 | 6144 | 281.75 | 158.21 | 1.78x |
| 1536 | 6144 | 2048 | 362.41 | 162.52 | 2.23x |
| 1536 | 2048 | 4096 | 172.50 | 109.70 | 1.57x |
| 1536 | 2048 | 2048 | 120.36 | 66.87 | 1.80x |

FP16 GEMM remains faster on all four shapes. The benchmark's exact
single-interval INT8 reference passes with zero tolerance. M=16 and M=128
latencies also improve across these four K/N geometries; M=1 continues
using the generic path.

For grouped weights, M/K/N=1536/1024/1024 and block=128, with five warmups,
20 repetitions and three trials:

| Weight bits | Remeasured per-token baseline (us) | Pipelined per-token (us) | Pipelined per-block A128 (us) | Pipelined per-tensor (us) |
|-------------|-----------------------------------|-------------------------|-----------------------------|-------------------------|
| 2 | 224.64 | 69.61 | 86.96 | 75.74 |
| 4 | 236.79 | 70.12 | 87.54 | 76.48 |
| 8 | 215.14 | 67.14 | 84.71 | 72.73 |

Per-token grouped execution improves 3.2–3.4x. Expanded-weight scratch,
prepack memory and global INT32 output remain zero. The activation scratch
budget and serialized input representation are unchanged. Weight-only
MatMulNBits remains faster, with different activation semantics.

The optimization overlaps tile copies with MMA, loads matrix fragments
directly from shared memory, decodes four low-bit codes together, and
reuses row/column scales across accumulator fragments. Exceptional scaling
uses an out-of-line helper that preserves the original INT32-to-FP32
rounding and gradual-underflow behavior.

Validation: 17 focused C++ tests pass in both standard and plugin CUDA
builds; nine Python tests plus 36 subtests pass, including replay with
changed inputs and concurrent execution of both pipelined paths. Tests
compare exact output bits for FP16/BF16, signed/zero scales, nonfinite
activations, extreme scales, unequal scale boundaries, packed-code ranges,
and partial tiles. Integrated Compute Sanitizer reports zero memory errors
and zero race hazards. These local results do not establish performance
or runtime portability on other GPU architectures.

### 8.6 Profile-guided grouped-kernel tuning

Measurements use the same H200 and source-build environment as section 8.5.
Before/after per-token measurements were taken on the same GPU with the
same benchmark inputs, 10 warmups, 50 repetitions, and five trials. These
are complete operator CUDA-event medians, including quantization and
host/launch gaps, not isolated GEMM timings.

For M/K/N=1536/1024/1024, weight block=128, constant scales, FP16 A/Y,
and no bias:

| Weight bits | Remeasured pipelined baseline (us) | Profile-guided operator (us) | Speedup |
|-------------|-----------------------------------|------------------------------|---------|
| 2 | 69.15 | 54.72 | 1.26x |
| 4 | 69.27 | 53.20 | 1.30x |

Final measurements for all supported widths and activation modes:

| Weight bits | Per-token (us) | Per-block A128 (us) | Per-tensor (us) |
|-------------|----------------|---------------------|----------------|
| 2 | 54.72 | 72.44 | 60.86 |
| 4 | 53.20 | 71.72 | 59.72 |
| 8 | 52.64 | 71.56 | 58.67 |

FP16 GEMM remains faster, at approximately 8 us for this geometry.
It does not have equivalent dynamic activation semantics.

Nsight Systems confirms `NativeAlignedGroupedMatMul<half, 4, 64>`.
For the per-token W4 case above, native GEMM decreases from 50.61 to
34.94 us; activation quantization is essentially unchanged at 6.74
versus 6.71 us. Nsight Compute identifies underutilized compute pipelines,
low eligible-warp counts, and register pressure rather than DRAM
saturation. A matched grouped test geometry, M/K/N=961/192/961 with
weight block=128 and activation block=64, reduces W4 register use from
166 to 135 per thread and kernel duration from 20.42 to 15.68 us.

The retained changes:

- Advance common scale intervals with independent weight/activation group
  counters instead of repeated integer division and remainder operations.
  Interval order and INT32 accumulator reset boundaries are unchanged.
- Cache activation scales until their group changes and distribute
  row/column scales with warp shuffles instead of redundant global loads.
- Use three asynchronous staging buffers for grouped packed weights/A.
- Expand W4 bytes with a byte permutation and two nibble masks, retaining
  the same centered signed codes.
- Specialize repeated FP16 scale products without the BF16 exceptional
  first-product path. Finite FP16 activation scales lie between
  `2^-24 / 127` and `65504 / 127`; multiplying a nonzero INT32 accumulator
  by such a scale always produces a normal finite FP32 first product.
  NaN propagation, the second multiplication, and FP32 rounding order
  are preserved. BF16 retains the gradual-underflow/overflow helper.
  Single-interval W8 retains its original low-register scaling code.

The kernels still execute `mma.sync.m16n8k32`, use `ldmatrix` for A
fragments (and B on the single-interval W8 path), and fuse scaling,
bias, and FP16/BF16 conversion into the output kernel. Grouped weights
must scale each common K interval before adding its FP32 register
contribution; a single per-channel epilogue is valid only when there
is one common interval. Neither path writes a global INT32 intermediate
or launches a separate dequantization pass. Expanded-weight scratch
and prepack memory remain zero.

Three-stage 128x128x64 and 128x128x128 single-interval tiles, cooperative
shared-weight expansion, wider grouped tiles, and maximum shared-memory
carveout were tested but not retained because they did not improve the
measured cases. The four W8 single-interval prefill cases in section 8.5
remain essentially unchanged at 159.46, 163.26, 110.24, and 66.59 us.
This is H200 evidence, not an Ada performance claim.

Validation: 18 focused C++ tests pass in both standard and plugin CUDA
builds. Ten Python tests pass, including capture/replay, concurrency,
initializer compatibility, and 18 new randomized exact-bit comparisons
covering W2/W4/W8, all activation modes, unequal scale boundaries, and
partial/small/large grouped tiles. New C++ cases cover the smallest
subnormal and largest finite FP16 activations, nonfinite domains, and
extreme signed/zero weight scales. Integrated Compute Sanitizer reports
zero memory errors and zero race hazards. Standalone SM75/80/90
compilation, changed-line C++ formatting, and Python lint/format checks
pass. Runtime behavior on Ada/Blackwell, Windows, and model-level
accuracy/performance remain unverified.

### 8.7 Register pipelining and common-interval specialization

The follow-up retains register-fragment ping-pong and a specialized grouped
path for weight block=128 with one activation scale interval. The latter
requires K divisible by 128, N divisible by 64, and complete M tiles
(32 or 64 rows according to the existing tile-selection rule). Other
geometry continues through the general grouped or fallback paths.
Neither optimization changes the selected activation quantization domain.

The same H200, FP16 inputs/output, constant weights/scales, no bias,
10 warmups, 50 repetitions and five trials as section 8.6 were used.
For K=N=1024 and weight block=128:

| Weight bits | M | Remeasured section 8.6 baseline (us) | Follow-up per-token operator (us) | Speedup |
|-------------|---|-------------------------------------|----------------------------------|---------|
| 2 | 128 | 34.40 | 29.85 | 1.15x |
| 2 | 1536 | 54.69 | 45.51 | 1.20x |
| 4 | 128 | 33.79 | 27.20 | 1.24x |
| 4 | 1536 | 53.22 | 43.22 | 1.23x |

These are complete operator CUDA-event medians, not bare GEMM timings.
Register ping-pong alone improved the measured per-token cases by about
1-2%; most of the improvement comes from eliminating dynamic interval
tracking, repeated activation-scale checks, and output-tail checks when
the specialized path's full-tile conditions hold. INT32 accumulators
still reset every 128 K elements, and FP32 scale products and interval
additions keep their original order.

Nsight Systems confirms `NativeAlignedGroupedMatMul<half, 4, 64, true>`.
For M/K/N=1536/1024/1024, native GEMM is 24.82 us versus 34.94 us in
section 8.6, while activation quantization remains 6.72 us.
The four W8 single-interval prefill cases remain essentially unchanged
at 159.31, 162.67, 109.96 and 66.65 us. FP16 GEMM remains faster.

A lossless lane-major W2/W4 weight layout and group-major scale layout
were also implemented and tested. They preserve all packed codes and
the raw FP16/BF16/FP32 scale bits; initialization produces additional
packed weights rather than an expanded INT8 matrix. Exact constant-weight
and fallback tests pass. However, neither tested layout version improves
the measured cases, so production prepacking was not retained.
For W4, M=1536, K=N=4096, the packed layout takes 333.53 us versus
324.20 us with the standard format and runtime weights. The benchmark's
`--runtime-weights` option permits reproducing the no-prepack comparison
with byte-identical weight input. Expanded-weight scratch, prepack memory,
and global INT32 output remain zero in the retained implementation.

Validation: 21 focused C++ tests pass in both standard and plugin builds.
Ten Python tests pass, including 36 randomized exact-bit cases,
capture/replay with changed inputs on the common-interval path, and
concurrent execution. New C++ coverage checks constant weights, both
tile sizes, FP16/BF16, unequal boundaries, tails, nonfinite activations,
and extreme scales on the specialized path. Integrated Compute Sanitizer
reports zero memory errors and zero race hazards. SM75/80/90 compilation
and changed-line C++/Python formatting and lint checks pass.
These results do not establish Ada/Blackwell runtime performance.

### 8.8 Standalone Hopper W8 prototype

A separate SM90a CUTLASS/CuTe prototype uses a strict fused INT8 GEMM
epilogue and a CuTe-AOT per-token CUDA quantizer. The quantizer runs
256 threads per row, implements rounded FP32 division and ties-to-even
INT8 conversion, and emits NaN scales/all-zero codes for nonfinite rows.
The timed path executes one quantization kernel and one GEMM kernel;
it contains no eager PyTorch quantization operations or allocations.

On H200, with 15 warmed samples of 100 repetitions, CUDA-event medians
in native C++ launch loops are:

| M | K | N | Selected tile | Quantize (us) | Bare GEMM (us) | Quantize + GEMM (us) |
|---|---|---|---------------|---------------|----------------|---------------------|
| 1536 | 2048 | 6144 | 128x128x128 | 8.17 | 57.04 | 66.67 |
| 1536 | 6144 | 2048 | 64x128x128 | 18.54 | 57.63 | 78.93 |
| 1536 | 2048 | 4096 | 128x128x128 | 8.17 | 35.94 | 44.34 |
| 1536 | 2048 | 2048 | 64x128x128 | 8.17 | 22.65 | 30.74 |

For M=128, the smaller tile wins among the two tested configurations,
with quantize+GEMM medians of 14.28-29.73 us across these K/N geometries.
These are prototype native-launch measurements, not integrated ORT
operator timings. They exclude allocations, transfers, constant weight
preparation, TMA descriptor setup, and warmup, and must not be interpreted
as direct end-to-end ORT speedup ratios against the Python I/O-binding
measurements in sections 8.4-8.7.

The prototype prepares signed INT8 B[N,K] losslessly from offset-packed
W8 and FP32 column scales before timing. Its CUDA quantization scratch
is qA plus row scales, at most 9,443,328 bytes for the measured shapes.
CUTLASS workspace is zero. The strict epilogue applies activation and
weight scales in the specified order, adds bias separately, and writes
FP16 directly without a global INT32 output or separate dequantization.
Binary checks report no fused multiply-add instructions.

An independent CPU integer-dot oracle verifies 47,710,208 main output
encodings, including quantized activation codes and FP32 scales.
Additional quantizer checks cover all 63,488 finite FP16 encodings,
rounding ties, NaN/Inf domains, signaling NaNs, signed zeros, and extreme
finite weight scales. This prototype validates FP16/per-token only;
BF16 and native per-tensor CUDA quantization remain unvalidated.

The reusable handle holds fixed shapes and qA/scale/weight/output pointers,
while raw A may change. GEMM/TMA initialization costs 2.12-2.34 us once;
complete handle creation costs 15.43-17.00 us and stores 1,792 host bytes
of kernel parameters. Run does not initialize descriptors. Image
availability is checked against the actual SM90a GEMM and quantizer
kernels, not just the device capability.

The prototype is not wired into the production operator. ORT integration
must handle varying output/scratch bindings, independent concurrent runs,
capture/replay lifetimes, per-device state, and constant signed-weight
storage before adopting the measured fast path. A ptxas C7510 warning
about WGMMA serialization across a function boundary remains in the
measured build and is another tuning opportunity.

### 8.9 Integrated Hopper W8 fast path

The SM90a CUTLASS fast path is now integrated into the ORT operator for
FP16 or BF16 A/Y (section 8.12) with constant W8 weights and one common scale interval.
It uses TMA warp-specialized ping-pong, INT8 WGMMA, a persistent tile
scheduler, and a strict fused epilogue. Scale storage may be FP16,
BF16, or FP32, and bias may be absent or have the activation type. The epilogue preserves
INT32-to-FP32 conversion, activation-scale multiplication, weight-scale
multiplication, and separate bias addition before output conversion.
Nonfinite activation domains retain their NaN output semantics.
No global INT32 output or separate dequantization pass is introduced.

Eligibility requires a built SM90a kernel image on a CC9.0 device,
constant standard W8 B with a single block, K divisible by 16 and at
least 128, N divisible by 8 and at least 512, and M at least 16.
The measured crossover additionally requires either K/N at least 1024
or M at least 128. Device pointers used by the GEMM must meet its
16-byte alignment requirements. A per-block activation mode can use
the path only when its one block covers all K. Runtime weights,
multiple scale intervals, smaller geometries, and
unsupported builds/images retain existing dispatch. Invalid mutable
scales still fail validation rather than being silently accepted.
Set `ORT_DISABLE_DYNAMIC_QUANTIZED_SM90_GEMM=1` before session creation
to retain the previous path for comparison.

Standard and plugin builds share SM90a source extraction and strict
floating-point compile flags. Unsupported architectures, CUDA-minimal
builds, and MSVC use a no-native availability stub, avoiding compilation
of unsupported TMA launchers. An actual GEMM initialization or launch
failure is returned to ORT, not hidden by retrying another kernel.

Before/after measurements below use the same H200, source build, FP16
A/Y, constant weights/scales, absent bias, stream, benchmark input,
10 warmups, 50 repetitions, and five trials. M=1536; weight block is
2048 except K=6144, which uses 8192. Values are complete ORT operator
CUDA-event medians including quantization, launch/host gaps, and the
per-call GEMM parameter setup, not standalone native-loop measurements.

| K | N | Previous per-token ORT (us) | Hopper per-token ORT (us) | Speedup | FP16 GEMM (us) |
|---|---|----------------------------|--------------------------|---------|----------------|
| 2048 | 6144 | 159.35 | 56.09 | 2.84x | 55.83 |
| 6144 | 2048 | 163.06 | 80.50 | 2.03x | 54.72 |
| 2048 | 4096 | 109.65 | 41.99 | 2.61x | 36.95 |
| 2048 | 2048 | 66.75 | 34.76 | 1.92x | 19.90 |

Per-tensor activation quantization improves from 168.25/170.48/116.84/73.50
to 63.19/87.38/49.30/41.60 us for the same ordered K/N cases.
M=16/128 cases also improve for these geometries. Small K/N crossover
measurements identified regressions for M=16/32, K=128, N=128 and
K=256, N=512; the retained gates exclude those cases.
FP16 GEMM remains faster, although the largest measured per-token case
is now close. The baselines do not have equivalent activation semantics.

Nsight Systems confirms the INT8 SM90 TMA/WGMMA kernel. At
M/K/N=1536/2048/6144 it averages 30.49 us; activation quantization
averages 12.97 us. Nsight Compute reports 32.22 us kernel duration,
62.76% compute throughput and 11.55% DRAM throughput for its profiled
run. These counters are not used as end-to-end timing measurements.

Initialization makes a lossless signed INT8 copy of constant W8 bytes
by XORing their symmetric offset. It retains the serialized weights
for fallback geometry, so this adds 12/16/8/4 MiB of persistent weight
memory in the four cases above, including serialized block padding.
W2/W4 and block-128 W8 use the grouped path of section 8.13, which keeps its own signed copy. Initialization
finishes on the per-thread default stream before the copy is published.
Quantized activation scratch remains within the existing row-tile
budget and CUTLASS workspace is zero.

Each Run prepares stack-local TMA/GEMM parameters for its current
scratch and output bindings. There is no shared mutable descriptor
cache, fixed-output assumption, device-wide synchronization, or
additional hot-path allocation. The readonly signed weights are
shared across runs. Capture/replay, changed activation values, output
rebinding, concurrent dynamic shapes, and mutable scale validation
are covered by session tests.

Validation: 23 focused C++ tests pass in both standard and plugin
builds, and 14 Python tests pass, including 12 new randomized
single-interval exact-bit cases across all three scale storage types
and per-token/per-tensor modes. Tests also cover both tile sizes,
padding, bias, nonfinite domains, extreme scales, and fallback geometry.
Integrated Compute Sanitizer reports zero memory errors and zero race
hazards. Generated SASS for all six tile/scale-type variants contains
no FFMA instructions. No-native SM75/SM80 stubs compile. Windows,
Ada/Blackwell execution and model-level performance remain unverified.

### 8.10 Register-resident activation quantizer

After the Hopper path, activation quantization was the largest non-GEMM
kernel (12.97 us at M/K=1536/2048). The scalar quantizer read each domain
twice from global memory with one thread per element. The replacement
(`QuantizeRowsVector`) is used when `K % 8 == 0`, the activation block is a
multiple of 8 (or covers K), a domain has at most 8192 elements, and the
activation and scratch pointers are 16/8-byte aligned. A group of 32 to 256
threads owns one domain, keeps its 16-byte loads in registers, reduces the
maximum with warp shuffles (plus shared memory for groups wider than one
warp), and writes eight codes per store. Other geometries use the unchanged
scalar kernel. The per-tensor scale is read directly and skips the
reduction.

The arithmetic is unchanged: nonfinite values mark the domain as infinite
before the order-independent maximum, the scale uses the same correctly
rounded division and denormal floor, and each code uses `__fdiv_rn`,
`__float2int_rn`, and the same clamp. NaN domains still produce NaN scales
and zero codes.

Quantizer kernel time, M=1536: 12.97 -> 7.1 us at K=2048 (group 256, one
chunk per thread) and 15.3 us at K=6144. Complete-operator H200 medians
(CUDA events, FP16 per-token, W8 Hopper path), before -> after:

| K/N | Before (us) | After (us) |
|-----|-------------|------------|
| 2048/6144 | 56.09 | 50.45 |
| 6144/2048 | 80.50 | 63.16 |
| 2048/4096 | 41.99 | 38.91 |
| 2048/2048 | 34.76 | 32.67 |

Per-tensor 2048/6144 improves from 63.19 to 59.82 us. The grouped W4/W2
M=1536, K=N=1024 per-token operator improves from 43.22 to 39.99 us and
45.51 to 42.84 us.

The remaining time between kernels is mostly host launch overhead. Building
the CUTLASS parameters costs about 2.7 us and launching the cluster kernel
about 6 us on the host. Building them before launching the quantizer did not
reduce end-to-end latency (the benchmark is host-bound), so it was not kept.

New coverage `RegisterResidentActivationQuantizer` exercises each group
width and chunk count, the 8192-element limit, the scalar fallback,
per-token, per-tensor, and per-block domains with tails, and infinite/NaN
rows against the exact CPU reference. Compute Sanitizer reports zero memory
errors and zero race hazards on the full operator test set.

### 8.11 CUDA graph readiness

The operator is CUDA-graph safe on every native path: all launches use the
compute stream, scratch comes from the framework allocator, the CUTLASS
parameters are rebuilt on the stack for each run so a replay never depends
on a stale host descriptor, and there is no host synchronization or
allocation in `Compute`. Constant, non-overridable scales are required for
capture (section 5.1). Signed-weight preparation happens once in `PrePack`.

`test_capture_replays_match_exact_reference_for_native_paths` captures a
graph and replays it with fresh random activations (including a zero row),
comparing every output bit with the exact reference. It covers the Hopper W8
path (both tile sizes, per-token and per-tensor), tensor-core W8/W4/W2, and
per-block activation intervals, with both constant and runtime weights.

`benchmark_dynamic_quantize_matmul_nbits.py --cuda-graph` replays the ORT
graph. A trace shows the quantizer-to-GEMM gap falling from about 4.5 us to
0.29 us. Complete-operator H200 medians, FP16 per-token, M=1536,
non-graph -> graph: W8 K/N 2048/6144 50.45 -> 46.26 us, 6144/2048
63.16 -> 61.15 us, 2048/4096 38.91 -> 32.87 us, 2048/2048 32.67 -> 27.79 us;
W4 K=N=1024 39.99 -> 36.48 us; W2 42.84 -> 37.91 us. At M=16, K/N=2048/6144
it falls from 27.67 to 19.04 us. The remaining inter-replay time is ORT
per-run overhead outside the operator.

### 8.12 BF16 activations on the Hopper path

BF16 activations now use the same SM90a kernel. Only the epilogue differs:
BF16 activation scales span the FP32 range, so INT32-accumulator times
scale can overflow or become subnormal, and the operator contract forms such
products in FP64 from the original inputs. The first two multiplications are
a single custom CUTLASS epilogue node (`StrictScaleProductOp`) over the
accumulator, activation scale, and weight scale. For a fragment whose scales
all satisfy 2^-126 <= |scale| < 2^96 (so any nonzero accumulator, at most
2^31, gives a normal finite product), the node issues the two plain
multiplies. Otherwise it calls a non-inlined routine with the full rule.
The range test is one integer subtract and compare per element, evaluated as
one branch per fragment. Per-element variants cost 9-31 us extra (an inlined or
called slow path, or always computing the FP64 result, were all slower);
the fragment-level test leaves GEMM time at 30.9 us versus 30.5 us for the
unchecked FP16 tree at M/K/N=1536/2048/6144.

`HopperBFloat16Domains` compares bitwise against the exact reference with
rows spanning subnormal, ordinary, near-overflow, NaN and infinite
activations, zero/denormal/maximum weight scales, all three scale storage
types, bias, per-token and per-tensor modes, and a large tile case.
Removing the slow path makes it fail. Because the node needs no FP16-only
assumption, the constructor no longer inspects the input type.

H200, BF16, M=1536, per-token, complete operator, previous path -> Hopper
(FP16 FP GEMM reference in parentheses): K/N 2048/6144 151.76 -> 49.62
(53.65) us, 6144/2048 145.97 -> 63.52 (52.31), 2048/4096 103.82 -> 39.70
(35.69), 2048/2048 62.73 -> 32.66 (19.24). Standard and plugin builds pass
25 focused tests each, the 15 Python tests pass, and Compute Sanitizer
reports zero memcheck errors and zero racecheck hazards.

### 8.13 Hopper grouped-scale path for W2/W4/W8

The remaining gap for W2/W4 (about 40 us against 8 us for FP16 at M=1536,
K=N=1024) came from the `mma.sync` grouped kernel running at roughly 7%
of INT8 peak. A standalone prototype (kernel authored in a separate
session, CuTe/CUTLASS 3 with hand-scheduled warp specialization) is now
integrated into `dynamic_quantize_matmul_nbits_sm90.cu`.

Design (first iteration): weight block exactly 128 (one K tile per scale
block), K a multiple of 128, one activation interval covering K, constant
weights. Section 8.14 generalizes these limits.
One producer warpgroup issues TMA for A and B and gathers the native
`[N, G]` weight scales (any of FP32/FP16/BF16 storage), and BM/64 consumer
warpgroups issue INT8 WGMMA. Every K tile accumulates into a fresh INT32
accumulator that is promoted to the FP32 running sum as
`sum += (float(acc) * a_scale) * b_scale`, in K order with no FMA (the
running sum starts at -0.0f so the first term needs no special case). Two
accumulators let the MMA of tile k+1 overlap the promotion of tile k.
There is no split-K because it would reassociate the sum. The epilogue adds
the optional bias, converts to FP16 or BF16, and stores 16 bytes per
thread through shared memory. BF16 activations reuse the section 8.12
scale-range test: rows whose activation scale leaves the FP32 normal range
take the FP64 first-product rule. Tile choice is 64x32 for at most 128
rows or N <= 512, 128x96 for N <= 1024, otherwise 128x128, all with four
stages; there is no workspace and the TMA descriptors are built per run.

Weights: `PrePack` expands the codes to centered signed INT8 `[N, K]`
(W4 code - 8, W2 code - 2, W8 XOR 0x80); this is lossless but costs N*K
bytes of persistent memory in addition to the serialized weights (2x the
packed size for W4 and 4x for W2). Use `ORT_DISABLE_DYNAMIC_QUANTIZED_SM90_GEMM=1`
to avoid it. The benchmark reports the bound as `prepack_upper_bound_bytes`.
Eligibility also needs N a multiple of 8 and at least 256, and 16-byte
aligned output and bias. Any row count is accepted; TMA zero-fills missing rows.

H200, FP16 per-token, previous path -> Hopper grouped (M, K, N, bits):

| Shape | Previous (us) | Hopper (us) |
|-------|---------------|-------------|
| 1536, 1024, 1024, W4 | 39.91 | 23.65 |
| 1536, 1024, 1024, W2 | 42.13 | 23.50 |
| 1536, 4096, 4096, W4 | 314.19 | 100.84 |
| 128, 4096, 4096, W4 | 66.34 | 25.77 |
| 16, 4096, 4096, W4 | 758.67 | 28.27 |
| 1536, 1024, 1024, W4 BF16 | 51.50 | 35.62 |

With CUDA graph replay the first row is 19.4 us. The GEMM kernel itself
takes about 8-9 us in the first case (24.9 us before); the rest is
activation quantization and launches. `HopperGroupedScaleIntervals` and
`HopperGroupedExceptionalDomains` compare bitwise with the exact reference
across all widths, modes, activation and scale types, tile configurations,
ragged shapes, bias, and NaN/Inf/subnormal/near-overflow domains. The
capture test covers a W4 constant-weight graph. Memcheck reports no errors.
Racecheck reports hazards on the scale-staging stores of the 64x32 tile; it
does not model the mbarrier release/acquire that orders them (the stores
precede a `__syncwarp` and a `mbarrier.arrive` that carries the compiler
memory clobber, and consumers read after the barrier wait), and results
are exact across all tests. The same tools report nothing for the TMA-only
W8 CUTLASS kernel.

Persistent scheduling and dispatch (second iteration). A persistent variant
(`PersistentGroupedKernel`, 128x128 tile, five stages) is used for the large
shapes: CTAs walk output tiles round-robin by `blockIdx.z` (no atomics or
global state, so it is CUDA-graph safe), the shared-memory stage ring runs
across tiles so the producer prefetches the next tile while the consumers
promote the current one, the scale-gather warp keeps the next K tile's
scales in registers, and promotion reads scales in chunks of four column
groups, which also removes the ptxas C7517 injected-wait warning. The
numerics are unchanged. TMA multicast with 2x1, 1x2 and 2x2 clusters was
prototyped and gave a geometric-mean 0.96-0.98x versus no cluster, so it is
not used; ping-pong consumers and 128x192/192x128 tiles did not help
(register spills, and the GPU is at its power cap with MMA and exact
promotion costs close to additive). Dispatch: at most 128 rows with
N <= 4096 uses 64x32; at most 128 rows with larger N uses 64x64; at most
512 rows with N <= 1024 uses 64x32; at most 2048 rows with N <= 1024 uses
128x96; everything else uses the persistent 128x128 kernel. Thresholds are
tuned on H200 (132 SMs) and must be re-tuned for other SM counts.

Two defects found while measuring the BF16 path. Rows beyond M in a tile used
scale 0.0, which fails the BF16 plain-range test and sent whole warps through
the FP64 rule; padded rows now use 1.0 (their outputs are never stored), and
the plain/slow choice is one warp-uniform branch outside the element loops.
The grouped path also previously rejected fewer than 16 rows, which left the
old small-M kernel (about 700 us at K=N=4096, M=16) in use for decode and
for a 4-row remainder tile when M exceeds the activation scratch budget; all
row counts now use the Hopper kernel.

H200 complete-operator medians, W4, block 128, per-token:

| M, K, N | dtype | Before (us) | After (us) |
|---------|-------|-------------|------------|
| 1536, 4096, 4096 | FP16 | 100.84 | 95.11 |
| 4096, 4096, 4096 | FP16 | 436.65 | 249.52 |
| 1536, 4096, 4096 | BF16 | 252 | 114.1 |
| 1536, 1024, 1024 | BF16 | 35.62 | 24.74 |
| 1 / 16 / 128, 4096, 4096 | BF16 | 78 | 30-32 |
| 1 / 16 / 128, 4096, 4096 | FP16 | 31 / 28 / 28 | 32 / 31 / 28 |
| 256, 1024, 1024 | FP16 | n/a | 20.21 |

BF16 remains 10-20% slower than FP16 on the large persistent shapes because
of the range test and BF16 scale conversion; the plain-multiply BF16 variant
is within about 10% of FP16.

### 8.14 Weight blocks 32-4096+ and several activation intervals on Hopper

The grouped kernel now handles any power-of-two weight block of at least 32,
K any multiple of 16, N a multiple of 8 and at least 256, and per-block
activation modes whose activation block is at least K or a multiple of 32.
Weight block 16 and activation blocks that are not multiples of 32 (including
below 32) keep the `mma.sync` kernels, since INT8 WGMMA is k32. The design was
prototyped separately and validated against an extended exact CPU reference
(129,792 bitwise comparisons, no mismatches) before integration.

An interval is a maximal K range with one weight block and one activation
interval. The K tile BK is the largest of 128, 64 and 32 that divides the
weight block and, with several activation intervals, the activation block,
so every interval boundary is a tile boundary (SW128/SW64/SW32 swizzle for the
three tile widths; a partial last tile is zero-filled by TMA, and zeros add
exactly 0 to the INT32 accumulator). An interval of R tiles accumulates in one
INT32 register accumulator (`ScaleOut::Zero` on its first tile), and two
accumulators alternate between consecutive intervals. After each tile is
issued, `warpgroup_wait<1>` retires the previous one and releases its shared-
memory stage immediately, so the producer never waits behind a promotion. The
accumulator is promoted when its interval ends, with the same
`sum += (float(acc) * a_scale) * b_scale` rule. Activation scales `[M, Ga]` are
read per interval from global memory (one load when a single interval covers
K). Weight scales keep the native `[N, G]` layout and the same signed INT8
expansion as before; the row stride is `G * block`, so padded last blocks cost
N * G * block bytes. Variants are the weight block equal to BK (one tile per
interval, no bookkeeping) or a multiple of BK, for BK 128/64/32, with the
64x32, 128x96 and (BK 128 only) 128x128 tiles; the persistent kernel still
serves weight block 128 with one activation interval. Tile choice compares CTA
counts with the SM count. Intervals that start inside a tile ("split" variants
for activation blocks such as 96) were prototyped but are not integrated:
they were faster only in a few cases (8-22%) and need extra kernels.

H200, M=1536, K=N=4096, W4, FP16 per-token unless noted, complete operator,
previous `mma.sync` path -> Hopper:

| Geometry | Previous (us) | Hopper (us) |
|----------|---------------|-------------|
| weight block 128, per-block activation 128 | 495.8 | 112.1 |
| weight block 64 | 666.4 | 159.1 |
| weight block 32 | 4705.8 | 248.5 |
| weight block 256 | 371.0 | 83.3 |
| weight block 4096 (one block per channel) | 282.2 | 66.7 |
| M=16, weight block 64 | 1491.9 | 30.8 |

Coarser blocks are faster than block 128 because promotions are amortized over
several tiles; finer blocks are slower (block 64 is 1.7x and block 32 3.1x the
block-128 time) because promotion work per 32 or 64 K dominates. The new
`HopperGroupedWeightBlocksAndActivationIntervals` test compares bitwise with the
exact reference across widths 2/4/8, weight blocks 32-512 and 4096, K tails,
activation blocks 24/32/96/192/256/1000 (including the unsupported ones that
fall back), per-token and per-tensor modes, bias, all scale types and both
activation types. All 28 focused tests pass in the standard and plugin builds,
the 15 Python tests pass, memcheck is clean, and the sm90 translation unit
still compiles as a stub for sm_89 and sm_100a. Ptxas reports no WGMMA
serialization warning for any grouped instantiation.

The Hopper-specific work did not cover Ada/Blackwell tuning, Windows
execution, or the split-interval variants. Section 8.15 covers the portable
kernel on SM120.

### 8.15 RTX 5090 portable-kernel tuning

Measured on an NVIDIA GeForce RTX 5090 (SM120), driver 610.88, CUDA 13.4,
PyTorch 2.14.1+cu130, and an ORT 1.32.0 Release CUDA-plugin build with native
SM120 SASS on Windows. The SM90 TMA implementation is intentionally a stub
under MSVC, so these measurements exercise the portable SM80+ `mma.sync`
kernels. The results were rechecked after commit `4a889f4722`, which
generalizes the SM90 WGMMA path but does not change SM120 dispatch.

The W8 single-interval TileK=64/128 alternatives and grouped TileM=32/64
alternatives were measured by alternating both candidates in one session
against the same inputs. Temporary per-kernel CUDA events exclude ORT and
host-launch gaps; every pair produced bit-identical output.

SM120 consistently prefers TileK=128 for the W8 single-interval kernel,
including row counts below the H200 crossover:

| M | K | N | TileK=64 (us) | TileK=128 (us) | K128 speedup |
|---|---|---|---------------|----------------|--------------|
| 16 | 2048 | 6144 | 21.09 | 18.80 | 1.12x |
| 32 | 2048 | 4096 | 42.58 | 40.30 | 1.06x |
| 64 | 6144 | 2048 | 68.66 | 61.01 | 1.13x |
| 1536 | 2048 | 4096 | 116.96 | 99.68 | 1.17x |

For grouped weights, TileM=32 remains better through 64 candidate tiles,
while TileM=64 wins at 128 or more. Representative K=N=4096 results:

| Bits | M | TileM=64 candidate tiles | TileM=32 (us) | TileM=64 (us) | M64 speedup |
|------|---|--------------------------|----------------|----------------|-------------|
| 2 | 64 | 64 | 47.18 | 52.93 | 0.89x |
| 2 | 128 | 128 | 54.75 | 52.75 | 1.04x |
| 4 | 64 | 64 | 25.78 | 31.82 | 0.81x |
| 4 | 128 | 128 | 59.34 | 53.02 | 1.12x |
| 8 | 64 | 64 | 27.63 | 31.25 | 0.88x |
| 8 | 128 | 128 | 44.03 | 35.50 | 1.24x |

Dispatch therefore uses TileK=128 for every eligible SM120 W8
single-interval shape and lowers the grouped large-tile threshold from 256
to 128 on SM120. Earlier architectures retain the H200 choices.

The other useful difference was the grouped kernel's minimum M: the kernel
already supports partial row tiles, but the portable dispatch required at
least 32 rows. Isolated kernel CUDA-event timings for block-128, FP16
per-token, K=N=4096 show the crossover clearly:

| Bits | M | Previous fallback (us) | Aligned grouped (us) | Speedup |
|------|---|------------------------|----------------------|---------|
| 2 | 8 | 198.30 | 48.90 | 4.06x |
| 2 | 16 | 372.19 | 30.08 | 12.37x |
| 4 | 8 | 218.11 | 52.18 | 4.18x |
| 4 | 16 | 380.14 | 30.30 | 12.55x |
| 8 | 8 | 182.69 | 35.74 | 5.11x |
| 8 | 16 | 343.52 | 29.98 | 11.46x |

At M=1 the result is mixed by width (0.75-1.18x), while all three widths
win at M=8. Dispatch therefore lowers the portable aligned-grouped minimum
from 32 to 8 only on SM120 or later; earlier architectures retain the
measured 32-row crossover.

Complete-operator CUDA-event medians after the change, with 20 warmups,
50 repetitions, seven trials, constant weights/scales, no bias, and the
same K=N=4096 geometry:

| Bits | M=1 (us) | M=8 (us) | M=16 (us) | M=32 (us) |
|------|----------|----------|-----------|-----------|
| 2 | 74.56 | 44.07 | 43.19 | 39.05 |
| 4 | 73.73 | 43.15 | 66.28 | 37.32 |
| 8 | 72.54 | 65.11 | 42.70 | 38.90 |

Windows host scheduling and GPU clock transitions make complete-operator
small-M timings visibly noisier than isolated kernel events, so the
per-kernel measurements determine the crossover. The grouped Python
reference test checks the M=8 path exactly for every weight width and
activation mode, and `BlackwellPortableKernelDispatch` adds exact FP16/BF16
coverage for the SM120 small-M, TileK=128, and grouped TileM=64 choices.

#### 8.15.1 MatMulNBits comparison on RTX 5090

A byte-identical weight-only comparison used FP16 A/Y, per-token activation
quantization, constant symmetric B/scales, K=N=4096, block size 128, no bias,
20 warmups, 100 repetitions, and nine trials. ORT CUDA Graph replay removes
most Windows host-launch variation. Each cell reports
`DynamicQuantizeMatMulNBits / MatMulNBits` CUDA-event median latency followed
by `MatMulNBits / DynamicQuantizeMatMulNBits` speedup:

| M | W2 us (speedup) | W4 us (speedup) | W8 us (speedup) |
|---|-----------------|-----------------|-----------------|
| 1 | 44.23 / 14.89 (0.34x) | 44.05 / 13.71 (0.31x) | 41.98 / 15.00 (0.36x) |
| 8 | 36.33 / 27.92 (0.77x) | 36.38 / 24.42 (0.67x) | 56.00 / 46.40 (0.83x) |
| 16 | 36.68 / 37.88 (1.03x) | 36.51 / 58.45 (1.60x) | 35.98 / 32.61 (0.91x) |
| 32 | 31.88 / 40.33 (1.27x) | 30.41 / 33.15 (1.09x) | 31.43 / 34.09 (1.08x) |
| 64 | 32.54 / 38.47 (1.18x) | 31.05 / 34.11 (1.10x) | 32.17 / 34.58 (1.08x) |
| 128 | 38.35 / 50.43 (1.32x) | 38.11 / 45.52 (1.19x) | 39.54 / 46.32 (1.17x) |
| 256 | 52.44 / 73.00 (1.39x) | 56.77 / 67.78 (1.19x) | 58.45 / 68.74 (1.18x) |
| 512 | 94.91 / 119.62 (1.26x) | 95.86 / 113.99 (1.19x) | 107.62 / 114.13 (1.06x) |
| 1536 | 227.30 / 270.45 (1.19x) | 247.67 / 267.54 (1.08x) | 266.60 / 267.62 (1.00x) |

The dynamic operator is not a decode replacement: MatMulNBits is 1.2-3.2x
faster at M=1/8. At block 128 the useful SM120 range begins around M=32,
where the dynamic operator wins for every tested width; its largest measured
gain is 1.39x for W2 at M=256. W8 becomes approximately tied by M=1536.
A host-inclusive W4 sweep over K/N of 2048/6144, 6144/2048, 2048/4096, and
2048/2048 won 15 of the 16 M=16/32/128/512 cases, confirming that the
mid-batch result is not limited to a square projection.

This plugin uses the compact fpA_intB MatMulNBits tactic set. Its optimized
path requires block size 32 for W4/W8 and 64 for W2, so enabling
`ORT_FPA_INTB_GEMM=1` does not change the block-128 baseline. A W4
K=N=4096, M=128, block-32 control did activate the profiled
CUTLASS non-Hopper GEMM with runtime-prepacked SM80 weights. Under the same
CUDA Graph protocol it measured 596.62 us for the dynamic operator and
37.89 us for MatMulNBits, making the dynamic operator 15.75x slower. The
standard-layout dynamic path is therefore not a replacement for optimized
MatMulNBits geometries.

The SM80+ W2 block-64, W4 block-32, and W8 block-32 paths also accept `weight_prepacked=1`, the existing
SM80 fpA_intB layout. It reads that layout directly, retains canonical
`[N, G]` scales, and preserves the same interval arithmetic as standard
DynamicQuantizeMatMulNBits (a W2 interval spans two 32-wide MMA steps accumulated
in INT32 before scaling). Eligibility requires K divisible by 128, N
divisible by 64, and one activation-scale interval. Fixed M=8/M=9 variants
serve drafting, while a 16-row tiled variant covers prefill. The measurements
below are for W4 only. Real BF16 qkv
weights from the Qwen drafter (K=5120, N=6144) measured:

| M | Standard-layout Dynamic (us) | Prepacked Dynamic (us) | Dynamic speedup | Prepacked MatMulNBits (us) |
|---|------------------------------|------------------------|-----------------|----------------------------|
| 8 | 1379.43 | 37.49 | 36.80x | 21.82 |
| 9 | 1540.54 | 64.42 | 23.91x | 36.27 |

Thus, prepacked dispatch is a substantial Dynamic operator optimization but
does not make activation quantization faster than the already memory-bound
fpA_intB MatMulNBits path. Model conversion remains an explicit accuracy and
performance decision rather than an automatic rewrite.

The performance gain also trades accuracy for activation quantization. Across
the block-128 matrix, relative RMSE versus decoded/scaled FP16 GEMM was about
0.8-1.0% for the dynamic operator and at most 0.12% for MatMulNBits. Model-level
accuracy remains a separate acceptance criterion. Large one-interval W8
weights are not a valid common benchmark in this checkout: MatMulNBits
terminates with integer divide-by-zero for the first K=2048, block=2048 case,
matching the limitation already observed on H200.


### 8.16 A100 (SM80) tuning of the portable kernels

The H200 choices left A100 slow in two regimes: fewer than 32 rows used the
one-warp-per-output fallback (1.2 ms at M=16, K=N=4096), and the 32x64/64x64
tiles with 64-code K tiles ran with few warps and short per-column weight
reads. Measured on an NVIDIA A100-SXM4-80GB (SM80, 108 SMs, 1410 MHz boost),
CUDA 13.0, PyTorch 2.12.1, a CUDA-plugin Release build with
`CMAKE_CUDA_ARCHITECTURES=80-real`, FP16 per-token activations, constant
weights/scales, no bias, ORT CUDA Graph replay and CUDA-event medians.

**Methodology.** An idle A100 starts at 210 MHz, and short kernels do not
raise the clock within the benchmark's default 10-20 warmups. A configuration
whose first measurements are slow therefore warms the GPU for the next ones,
which produced regressions that disappear with `--warmup 3000` (for example
M=32, identical kernels, 70 us versus 83 us). All numbers below use 3000
warmups, 100 repetitions, and the minimum of two interleaved old/new
processes. Run-to-run noise is still about 5%.

**Dispatch (SM80 only; SM90+ WGMMA, SM120 and other architectures unchanged).**

- The aligned grouped kernel is used for any M (previously at least 32).
- Tiles `16 x 32` (four warps over distinct columns), `16 x 64`, `32 x 64`
  and `64 x 64` are selected by `SelectGroupedTile`: the largest tile among
  `64 x 64`, `32 x 64` and `16 x 64` whose padded rows exceed M by at most
  30% and whose grid has at least 160 CTAs (about 1.5 per SM), otherwise
  `16 x 32`. On the 50 swept W4 points (M=16-512, N=1024-11008, K=4096) the
  selected tile is within 10% of the best of the four tiles with 128-code K
  tiles. At equal tile shape the 128-code K tile was slower than the 64-code
  tile (by more than 2%) at only one of the 200 swept combinations
  (`64 x 64`, M=96, N=11008), which the selector does not choose.
- A 128-code K tile (two stages for `64 x 64`, three otherwise) is used when
  K and the weight block are multiples of 128 and the activation block covers
  K or is a multiple of 128. Every promotion boundary then lies on a tile
  boundary, so the ordered FP32 accumulation is unchanged. The 128-byte
  activation rows and half as many K iterations gave 1.1-1.25x over 64-code
  tiles at equal tile shape.
- The fixed-geometry Common128 kernel also serves partial last row tiles by
  predicating rows. This was 1.3-2x faster than the generic kernel for ragged
  M such as 100, 200, 300 and 700, but 30% slower than the unpredicated kernel
  for aligned M, so both variants exist.
- Weight-scale loads are fetched one interval ahead (no measurable effect
  alone; it keeps the promotion off the global-load latency path).
- W8 weights with one block per channel use the grouped kernel below 640
  rows (1.09-1.17x over the single-interval tile at M=64-512; rows below 16
  previously had no tensor-core path) and the single-interval kernel with a
  128 K tile at 640 rows or more (4-6% faster than a 64 K tile at M >= 512).

Paired old/new, W4, block 128, K=N=4096 (complete operator, us):

| M | Previous | Tuned | Speedup |
|---|----------|-------|---------|
| 1 | 110.5 | 54.7 | 2.02x |
| 8 | 617.8 | 45.9 | 13.5x |
| 16 | 1205.0 | 44.3 | 27.2x |
| 32 | 70.5 | 48.3 | 1.46x |
| 48 | 106.3 | 55.4 | 1.92x |
| 64 | 79.3 | 62.8 | 1.26x |
| 96 | 79.7 | 73.1 | 1.09x |
| 128 | 100.5 | 89.4 | 1.12x |
| 192 | 120.6 | 90.6 | 1.33x |
| 256 | 129.2 | 113.0 | 1.14x |
| 300 | 178.4 | 134.8 | 1.32x |
| 512 | 207.3 | 176.1 | 1.18x |
| 700 | 447.9 | 322.9 | 1.39x |
| 1536 | 520.3 | 449.3 | 1.16x |

Across W2/W4/W8 and K/N of 1024/1024, 2048/6144, 4096/4096, 4096/11008,
6144/2048 and 11008/4096 (18 combinations), the geometric-mean speedup is
7.8x for M <= 16 (54 cases), 1.40x for M=32-64 (54), 1.23x for M=96-200 (90),
1.23x for M=256-700 (90), and 1.24x for M=1024-1536 (36). The sweep used a
25% padding threshold and had three W8 cases below 1.0x at M=100 (0.84-0.97x);
the final 30% threshold was rechecked only for W4/W8 at K/N of 4096/4096,
4096/11008 (W8) and 2048/6144 (W8), where M=100/200/300 measured 1.13-1.74x.
The other 105 sweep points were not rerun after that change.

The operator is still not a replacement for MatMulNBits on A100 above
about M=64. Block 128, K=N=4096, same protocol, DynamicQuantizeMatMulNBits
us / MatMulNBits us / FP16 GEMM us:

| M | W2 | W4 | W8 |
|---|----|----|----|
| 1 | 50.4 / 20.7 / 21.1 | 46.1 / 19.3 / 21.3 | 56.2 / 26.3 / 21.3 |
| 16 | 48.4 / 63.8 / 22.8 | 44.0 / 52.2 / 23.1 | 45.5 / 68.1 / 23.2 |
| 32 | 51.4 / 66.1 / 22.7 | 48.0 / 63.0 / 22.8 | 48.3 / 70.3 / 22.8 |
| 64 | 64.5 / 66.5 / 25.1 | 62.3 / 62.5 / 25.0 | 64.9 / 69.7 / 25.3 |
| 128 | 96.0 / 74.2 / 33.1 | 89.1 / 68.7 / 33.0 | 75.2 / 74.8 / 33.1 |
| 512 | 186.5 / 124.2 / 87.4 | 175.9 / 120.8 / 88.5 | 161.4 / 128.6 / 88.6 |
| 1536 | 483.1 / 263.2 / 219.3 | 449.1 / 261.5 / 222.0 | 389.3 / 266.9 / 219.4 |

The dynamic operator wins only at M=16-64 (1.0-1.5x over MatMulNBits) and
at M=8 for W8. At M=1536 it reaches about 115 INT8 TOPS, against 624 peak, and
the FP16 cuBLAS GEMM is 2x faster.

Findings that did not pay off and bound future work:

- Stage depth (2-5) and a single barrier per K tile were neutral, and
  128x64/128x32 CTA tiles gave at most 1.2x on some 128-aligned W2/W4 row
  counts but were slower for W8 and ragged M, so they were not adopted;
  large-M time is nearly independent of tile shape.
- Large-M throughput appears limited by the structure of the `mma.sync`
  promotion loop rather than by memory or tile shape (ablating MMA, decode,
  copies or promotion individually moved time by under 20%, though the
  kernel is sensitive to compiler scheduling and the ablations perturbed it).
  Reaching cuBLAS-class throughput needs a different kernel (for example a
  CUTLASS INT8 GEMM with in-kernel ordered promotion) and was not attempted.
- Nsight Compute counters were unavailable on this host
  (`ERR_NVGPUCTRPERM`), so no per-kernel stall attribution exists.
- Only SM80 is tuned; SM86/SM89 keep the H200 rule until measured. The
  CTA threshold of 160 assumes 108 SMs.
- N not divisible by the tile columns takes the generic kernel instead of the
  fixed-geometry one, and K not divisible by 128 keeps 64-code K tiles.

Validation: `AmpereTunedTileDispatch` and
`test_row_counts_across_grouped_tile_selection_match_exact_reference`
compare bitwise with the exact reference for M from 1 to 641, all widths,
per-tensor/token/block activation modes, weight blocks 64/128/256 and K tails;
an additional 840-case sweep (28 row counts, 10 geometries) also matched
exactly. The benchmark script validates bitwise equality for every timed
configuration. Use `--warmup 3000` on a cold GPU.

## 9. Delivery Plan and Acceptance

| Priority | Delivery | Required outcome |
|----------|----------|------------------|
| P0 | Complete schema and correctness baseline | All three MatMulNBits weight widths (8/4/2), finite signed/zero scales, all three activation modes, single/multiple common scale intervals, shape/input policies, and bounded generic CUDA paths. |
| P0 | Primary prefill fast path | Per-token A8 with per-channel W8; fused scaling/bias/conversion and verified tensor-core dispatch. |
| P0 | Integration and compatibility | Byte-identical symmetric offline-quantizer initializer reuse; explicit export opt-in; standard/plugin builds; shape/numerical/capture/concurrency tests and documentation. |
| P1 | INT4 and block-scaled native paths | Decode offset-packed W4 without a full expanded weight matrix; tune W4A8/W8A8 grouped dots and per-token K-block A8, starting with matched block sizes before unequal-boundary fast paths. |
| P2 | INT2 native-path tuning | Decode four offset codes per byte without a full expanded weight matrix; tune W2A8 grouped dots within the 16..256 weight-block contract, retaining exact generic coverage for unsupported native geometry. |
| P2 | Additional latency tuning | Measured small-M dispatch and per-tensor reduction tuning; neither may change activation domains, weight codes/scales, or interval ordering. |
| Follow-up | Additional fusions | Use section 6.1 priorities; new residual/activation/output-quantization semantics require explicit contracts, not overloaded v1 inputs. |

The first advertised version must have a working fallback for every
schema-supported weight width, weight/activation block size, and dynamic
mode, including INT2. INT2 correctness and offline-tool compatibility are
P0 v1 requirements even though native INT2 performance tuning is P2.
A preliminary PR may land an explicitly limited subset, but must
reject unsupported widths/modes/geometries instead of advertising complete
version-1 support. Optimization priority never authorizes a floating GEMM
fallback, approximate scale normalization, or silent weight repacking that
changes the decoded values.

Acceptance requires quantization-reference parity, semantics-preserving
dispatch, bounded scratch, capture/concurrency correctness under the documented
constant-scale capture restriction, both CUDA build
surfaces, and measured end-to-end benefit on the target prefill workloads.
The core design invariant is:

**The operator always dynamically quantizes activations. The model selects
the granularity; CUDA dispatch selects only an implementation of that contract.**

## 10. Source Precedents and Design Implications

The following conclusions come from local source snapshots, not performance
claims or a guarantee of exhaustive backend coverage. Links pin the inspected
revisions so changes upstream do not silently change the evidence.

| Repository snapshot | Relevant usage | Implication for this design |
|---------------------|----------------|-----------------------------|
| vLLM `16282a9c4` | [W8A8 CUTLASS][vllm-w8a8] uses dynamic per-token INT8, column weight scales, and optional bias; [block INT8][vllm-block] scales K-block dots in the mainloop. | Default to per-token; distinguish final-epilogue-only and interval-scaled paths. |
| TensorRT-LLM `89e1410d6c` | [QServe][trtllm-qserve] uses per-token INT8 and progressive INT4 weight quantization; [RMSNorm quantization][trtllm-rms] fuses a producer with dynamic quantization. | Integer reconstruction is a specialized format, not arbitrary floating block scales; producer-side fusion is valuable follow-up work. |
| SGLang `0da18f8d91` | [QoQ][sglang-qoq] combines dynamic per-token symmetric A8 with asymmetric channel/group W4; [block INT8][sglang-block] uses matched A/B K-group scales and FP32 partial accumulation. | Do not confuse weight per-channel naming with activation granularity; generic independent A/B block sizes are a deliberate extension. |
| llama.cpp `f3f1a8f27` | [CPU type traits][llama-traits] pair Q4/Q8 weights with block Q8 activations; [CUDA MMQ quantization][llama-mmq] uses format-dependent block scales and correction sums. | Per-block activation quantization is established practice, but GGUF block metadata/layouts are not the MatMulNBits ABI. |
| FlashInfer `fa2672fb` | [CUTLASS MoE instantiations][flashinfer-cutlass] include floating activations with integer weights and FP8 with INT4; [MXINT4 launch][flashinfer-mxint4] selects BF16 activations. | W4A8 naming can mean FP8, not INT8; these are layout/fusion precedents, not equivalent integer-activation kernels. |

vLLM's [Marlin scale preprocessing][vllm-marlin-scales] normalizes grouped
scales and rounds them into an integer representation. QServe similarly
reconstructs INT8-domain weights using progressive integer group parameters.
Neither may replace this operator's original MatMulNBits floating scales
or logical codes without an explicitly different format/accuracy contract.
Symmetric MatMulNBits quantization is chosen for offline-tool reuse, not
because every reference repository uses the same packed format.

vLLM has INT8-capable [RMSNorm/quantization][vllm-rms] and
[SiLU-and-multiply/block-quantization][vllm-silu] kernels. Kernel availability
does not establish that its compiler registers the same INT8 fusion.
llama.cpp's [quantized matvec epilogue][llama-glu] has bias/gate/GLU fusion,
with shape-dependent dispatch; it is not evidence of universal prefill fusion.
These sources motivate section 6.1 without broadening version 1's schema.

[vllm-w8a8]: https://github.com/vllm-project/vllm/blob/16282a9c4ee754bedd67f72b01ac76ae8568dbdd/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
[vllm-block]: https://github.com/vllm-project/vllm/blob/16282a9c4ee754bedd67f72b01ac76ae8568dbdd/vllm/model_executor/layers/quantization/utils/int8_utils.py
[vllm-marlin-scales]: https://github.com/vllm-project/vllm/blob/16282a9c4ee754bedd67f72b01ac76ae8568dbdd/vllm/model_executor/layers/quantization/utils/marlin_utils.py
[vllm-rms]: https://github.com/vllm-project/vllm/blob/16282a9c4ee754bedd67f72b01ac76ae8568dbdd/csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu
[vllm-silu]: https://github.com/vllm-project/vllm/blob/16282a9c4ee754bedd67f72b01ac76ae8568dbdd/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu
[trtllm-qserve]: https://github.com/NVIDIA/TensorRT-LLM/blob/89e1410d6cddaf4c49a89dec8541e9a6baf76aa5/tensorrt_llm/quantization/quantize.py
[trtllm-rms]: https://github.com/NVIDIA/TensorRT-LLM/blob/89e1410d6cddaf4c49a89dec8541e9a6baf76aa5/tensorrt_llm/quantization/functional.py
[sglang-qoq]: https://github.com/sgl-project/sglang/blob/0da18f8d916eb2d99b7382851d19426554a74cf3/python/sglang/srt/layers/quantization/qoq.py
[sglang-block]: https://github.com/sgl-project/sglang/blob/0da18f8d916eb2d99b7382851d19426554a74cf3/python/sglang/srt/layers/quantization/int8_kernel.py
[llama-traits]: https://github.com/ggml-org/llama.cpp/blob/f3f1a8f2760f28325a5ec20c05b171e5b7c83a29/ggml/src/ggml-cpu/ggml-cpu.c
[llama-mmq]: https://github.com/ggml-org/llama.cpp/blob/f3f1a8f2760f28325a5ec20c05b171e5b7c83a29/ggml/src/ggml-cuda/quantize.cu
[llama-glu]: https://github.com/ggml-org/llama.cpp/blob/f3f1a8f2760f28325a5ec20c05b171e5b7c83a29/ggml/src/ggml-cuda/mmvq.cu
[flashinfer-cutlass]: https://github.com/flashinfer-ai/flashinfer/blob/fa2672fbc76b41cc08478c39395d401b5ec6f98f/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu
[flashinfer-mxint4]: https://github.com/flashinfer-ai/flashinfer/blob/fa2672fbc76b41cc08478c39395d401b5ec6f98f/csrc/trtllm_fused_moe_kernel_launcher.cu
