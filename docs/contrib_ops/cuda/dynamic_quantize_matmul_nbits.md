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
  128 (section 8.16). On SM80 and SM89 it also accepts weight blocks of 32 and
  activation blocks that are multiples of 32, promoting after every 32-wide
  MMA step (sections 8.17 and 8.18). SM89 (RTX 4090) accepts any positive M:
  small-M and 32-aligned fine intervals use the four tile shapes above with
  a 192-CTA occupancy target. Coarse 64-aligned intervals at M >= 32 retain
  the established `32 x 64` / `64 x 64` selection.
  Compact SM86 and SM120 devices with at most 48 SMs also accept positive M
  and 32-aligned fine intervals (section 8.20). Compact SM86 small-M choices
  target three CTAs per SM; M >= 32 retains the established coarse-path
  tile selection. Compact SM120 uses all four shapes, balancing padded
  rows against a target of two CTAs per SM for W2 or M < 64, and one CTA
  per SM for W4/W8 at M >= 64.
- SM80-prepacked W2 block-64 and W4/W8 block-32: two-stage asynchronous
  staging, direct register decoding, and typed weight scales fetched ahead.
  A CTA covers 64 columns and one 16-row fragment by default. SM80 and SM89
  reuse decoded weights across two row fragments, or four for W2/W8, when
  the row count and grid permit it. Multi-fragment selection targets at
  least 96 CTAs on SM80 and 128 on SM89. Compact SM86/SM120 instead target
  `floor(4 * SM_count / 3)` CTAs, with up to four W2/W8 fragments, two W4
  fragments on SM86, and four W4 fragments on SM120. W2/W4 decoding produces
  both MMA registers from the same loaded word pair; W8 retains its original
  byte permutation. The schema's prepacked-layout eligibility checks still
  apply.
- Other native W8A8/W4A8/W2A8: `32 x 64 x 64` CTA tile with direct shared-memory
  weight decoding. Requires M at least 32, N at least 8, K at least 32,
  and every interior activation/weight block boundary 32-aligned: each
  block size must be at least K or divisible by 32.

M/N tails and final partial K tiles are predicated. Unsupported geometry
uses the one-warp-per-output DP4A/scalar path; all paths accumulate common
intervals in the same ordered FP32 sequence.

The single-interval M threshold was checked against DP4A on the measured
H200 shapes, but is not universally tuned. Aligned pipeline tile choices
were measured on H200; SM80 (A100), SM89 (RTX 4090), and SM120 have their own
measured choices (sections 8.15, 8.16, and 8.18). Compact SM86 (RTX 3060)
and SM120 (RTX 5060 Ti) measurements are in section 8.20; other architectures
remain unverified.
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

### 8.17 Prepacked weights and 32-aligned intervals on A100

Measured with `bench_prepacked.py` (kept outside the tree; it times
`DynamicQuantizeMatMulNBits` with standard blobs (`dyn_std`), the same operator with
`weight_prepacked=1` (`dyn_pre`), and prepacked MatMulNBits (`mnb_pre`)), commit
`f4a6469479` plus section 8.16, on an idle A100-SXM4-80GB, CUDA 13.0, FP16
per-token, CUDA Graph replay, 2000 warmups, 100 repetitions, median of 9 trials.
The first attempt was discarded: another user's job started on the same GPU
mid-run and produced ~2.2 ms time-slice plateaus, so check `nvidia-smi
--query-compute-apps` before and after a run.

`mnb_pre` could not run because this build has `onnxruntime_USE_FPA_INTB_GEMM=OFF`.
The baseline below is therefore standard-layout MatMulNBits (`mnb_std`, byte-identical
weights, no activation quantization) from `benchmark_dynamic_quantize_matmul_nbits.py`;
the real prepacked MatMulNBits is expected to be faster, especially at small M.
`dyn_pre` equalled `dyn_std` bit for bit in all 48 cases.

K=N=4096, microseconds:

| Weights | M | dyn_std | dyn_pre | mnb_std | dyn_std/dyn_pre | mnb_std/dyn_pre |
|---------|---|---------|---------|---------|-----------------|-----------------|
| W2 b64 | 1 | 76.0 | 43.0 | 21.1 | 1.77x | 0.49x |
| W2 b64 | 8 | 74.4 | 40.1 | 40.7 | 1.85x | 1.01x |
| W2 b64 | 16 | 77.9 | 43.5 | 64.1 | 1.79x | 1.47x |
| W2 b64 | 32 | 84.7 | 52.8 | 65.8 | 1.60x | 1.25x |
| W2 b64 | 64 | 116.6 | 63.8 | 65.5 | 1.83x | 1.03x |
| W2 b64 | 128 | 177.6 | 92.6 | 73.2 | 1.92x | 0.79x |
| W2 b64 | 512 | 467.9 | 273.8 | 127.7 | 1.71x | 0.47x |
| W4 b32 | 1 | 375.8 | 54.6 | 20.1 | 6.9x | 0.37x |
| W4 b32 | 8 | 2362.6 | 47.0 | 33.2 | 50x | 0.71x |
| W4 b32 | 16 | 4662.1 | 55.1 | 52.9 | 85x | 0.96x |
| W4 b32 | 32 | 821.7 | 68.0 | 63.5 | 12x | 0.93x |
| W4 b32 | 64 | 860.7 | 84.8 | 62.9 | 10x | 0.74x |
| W4 b32 | 128 | 908.5 | 131.1 | 69.2 | 6.9x | 0.53x |
| W4 b32 | 512 | 2702.7 | 426.0 | 125.8 | 6.3x | 0.30x |
| W8 b32 | 1 | 378.8 | 51.4 | 27.6 | 7.4x | 0.54x |
| W8 b32 | 8 | 2357.3 | 46.6 | 69.1 | 51x | 1.48x |
| W8 b32 | 16 | 4650.5 | 52.2 | 69.3 | 89x | 1.33x |
| W8 b32 | 32 | 864.6 | 66.2 | 71.4 | 13x | 1.08x |
| W8 b32 | 64 | 926.0 | 84.1 | 70.5 | 11x | 0.84x |
| W8 b32 | 128 | 1021.7 | 129.5 | 75.7 | 7.9x | 0.58x |
| W8 b32 | 512 | 3100.5 | 406.0 | 134.3 | 7.6x | 0.33x |

K=5120, N=6144 shows the same pattern: `dyn_pre` beats `mnb_std` at M=8-32 for
W8 (1.4-1.9x), M=16-64 for W2 (1.3-2.2x) and M=16-32 for W4 (1.2-1.4x), and
loses at M=1 and from M=64-128 (W4 M=512: 796 us against 201 us).

Findings:

- The large `dyn_std` gaps for W4/W8 block 32 exist because the aligned grouped
  kernel requires `block_size % 64 == 0`; block 32 falls to the older
  `NativeIntegerMatMul` (M >= 32) or the DP4A/scalar kernel (M < 32). W2 block
  64 already uses the grouped kernel, so its 1.7-2.0x prepacked gain is the
  layout itself.
- The prepacked kernel's fixed 16-row CTA tile re-reads the weights once per
  16 rows and has no larger tile, so it scales poorly: W4 K=N=4096 goes from
  55 us (M=16) to 131 us (M=128) and 426 us (M=512). The older `b1.json`/`b2.json`
  runs on another GPU show the same shape (M=512: 274 us against 101 us
  prepacked MatMulNBits).
- On A100 the useful window is roughly M=8-32 with 1.1-2x over standard
  MatMulNBits; at M=1 MatMulNBits is 2-3x faster.

#### 8.17.1 Changes after the baseline

Three changes followed, all SM80-only, bit-exact, and checked against the exact
reference and against each other (`dyn_pre == dyn_std` in every benchmark case):

1. **Prepacked row fragments.** `PrepackedMatMul` takes a `RowFragments`
   parameter: each warp keeps its eight columns and decodes a B fragment once for
   2 or 4 row fragments (CTA rows 32 or 64). `PrepackedRowFragments` picks the
   largest count that does not exceed M and leaves at least 96 CTAs (a CTA per
   SM), at most 2 for INT4 (whose nibble decode is the most expensive; four
   fragments were 0-10% slower than two at M=128-512) and 4 for INT2/INT8.
   The scales of the next 128-wide K tile are also loaded into registers ahead
   of the MMAs (neutral on its own at one fragment). Measured at K=N=4096,
   M=512: W2 288 -> 194 us, W4 423 -> 341 us, W8 407 -> 262 us; K=5120, N=6144,
   M=128: W2 162 -> 105 us, W4 238 -> 187 us, W8 232 -> 144 us. M <= 16 and
   the fixed M=8/9 kernels are unchanged.
2. **32-aligned intervals in the standard layout.** `NativeAlignedGroupedMatMul`
   has a `Fine` mode that zeroes and promotes the INT32 accumulators after every
   32-wide MMA step, so weight blocks of 32 and activation blocks that are
   multiples of 32 but not 64 (for example 96) use the grouped tensor-core
   kernel instead of `NativeIntegerMatMul` (M >= 32) or the DP4A kernel
   (M < 32). It is enabled only on SM80 and keeps the 64-code K tile and the
   existing tile selector. Weight scales of the whole tile are loaded before
   its MMAs. W4/W8 block 32, K=N=4096: M=16 4662 -> 115 us (40x), M=32 822 ->
   121 us, M=128 909 -> 248 us, M=512 2703 -> 776 us; K=5120, N=6144 gives
   62x at M=16 and 3.5-4x at M=512. Block-128 and block-64 results are unchanged
   within noise (paired old/new runs, section 8.16 protocol).
3. **Register cap for the runtime-interval kernel.** Restructuring the
   promotion into a shared lambda raised the register count of the 64x64
   runtime-interval instantiation from 151 to 174 and dropped it from three to
   two CTAs per SM (W2 block 64, M=512: 468 -> 645 us). Those instantiations
   now use `__launch_bounds__(128, 3)` (168 registers, 16 bytes of stack), which
   restored 458 us (1.02-1.04x over the previous code). The fixed-geometry
   Common128 instantiations are slower with any launch-bound or register
   attribute (6-7% at M=1536 even with `__maxnreg__(255)`), so the body lives in
   a `__forceinline__` function behind two kernels, one with and one without the
   attribute.

Complete-operator times (us) before and after on an idle A100, same protocol as
above (`mnb_std` is standard-layout MatMulNBits, not the prepacked kernel; M=1
differences of 10-15% are noise, as are M=32 differences between runs):

| K x N | Weights | M | dyn_std before | dyn_std after | dyn_pre before | dyn_pre after | mnb_std | mnb_std / dyn_pre |
|-------|---------|---|----------------|---------------|----------------|---------------|---------|-------------------|
| 4096x4096 | W2 b64 | 16 | 77.9 | 77.2 | 43.5 | 40.5 | 64.1 | 1.6x |
| 4096x4096 | W2 b64 | 64 | 116.6 | 111.9 | 63.8 | 60.6 | 65.5 | 1.1x |
| 4096x4096 | W2 b64 | 128 | 177.6 | 171.7 | 92.6 | 86.8 | 73.2 | 0.8x |
| 4096x4096 | W2 b64 | 512 | 467.9 | 457.8 | 273.8 | 193.8 | 127.7 | 0.7x |
| 4096x4096 | W4 b32 | 16 | 4662.1 | 115.0 | 55.1 | 51.1 | 52.9 | 1.0x |
| 4096x4096 | W4 b32 | 32 | 821.7 | 120.5 | 68.0 | 65.9 | 63.5 | 1.0x |
| 4096x4096 | W4 b32 | 64 | 860.7 | 159.0 | 84.8 | 84.0 | 62.9 | 0.7x |
| 4096x4096 | W4 b32 | 128 | 908.5 | 247.9 | 131.1 | 123.2 | 69.2 | 0.6x |
| 4096x4096 | W4 b32 | 256 | - | 397.8 | - | 185.5 | - | - |
| 4096x4096 | W4 b32 | 512 | 2702.7 | 775.8 | 426.0 | 340.6 | 125.8 | 0.4x |
| 4096x4096 | W8 b32 | 16 | 4650.5 | 110.7 | 52.2 | 50.3 | 69.3 | 1.4x |
| 4096x4096 | W8 b32 | 64 | 926.0 | 160.3 | 84.1 | 82.1 | 70.5 | 0.9x |
| 4096x4096 | W8 b32 | 128 | 1021.7 | 252.1 | 129.5 | 114.4 | 75.7 | 0.7x |
| 4096x4096 | W8 b32 | 256 | - | 398.1 | - | 170.7 | - | - |
| 4096x4096 | W8 b32 | 512 | 3100.5 | 775.3 | 406.0 | 262.4 | 134.3 | 0.5x |
| 5120x6144 | W2 b64 | 64 | 185.8 | 181.9 | 90.8 | 76.1 | 117.4 | 1.5x |
| 5120x6144 | W2 b64 | 128 | 282.7 | 275.7 | 156.7 | 105.1 | 120.8 | 1.1x |
| 5120x6144 | W2 b64 | 512 | 859.1 | 832.5 | 496.7 | 354.7 | 198.6 | 0.6x |
| 5120x6144 | W4 b32 | 16 | 8687.1 | 140.3 | 65.9 | 61.8 | 80.0 | 1.3x |
| 5120x6144 | W4 b32 | 64 | 1079.8 | 266.2 | 141.4 | 103.7 | 116.4 | 1.1x |
| 5120x6144 | W4 b32 | 128 | 1242.2 | 449.0 | 240.0 | 186.9 | 117.9 | 0.6x |
| 5120x6144 | W4 b32 | 512 | 4809.8 | 1374.0 | 796.4 | 631.3 | 201.5 | 0.3x |
| 5120x6144 | W8 b32 | 16 | 9066.9 | 139.9 | 69.3 | 64.8 | 120.5 | 1.9x |
| 5120x6144 | W8 b32 | 64 | 1432.7 | 272.6 | 128.6 | 103.8 | 124.6 | 1.2x |
| 5120x6144 | W8 b32 | 128 | 1450.4 | 442.3 | 230.4 | 144.2 | 127.5 | 0.9x |
| 5120x6144 | W8 b32 | 512 | 5599.8 | 1367.4 | 772.6 | 491.6 | 209.1 | 0.4x |

Analysis:

- The prepacked kernel and the fine standard kernel are both limited by the
  per-32-code promotion, not by memory: every output element needs an
  INT-to-float conversion, two ordered multiplies and an add per 32 K, about
  as many ALU instructions as the MMAs themselves, and bit-exactness forbids
  fusing or reordering them. A back-of-envelope count for M=512, K=N=4096 puts
  the promotion alone near 80 us of the 261-341 us (W4/W8) measured.
  Against standard-layout MatMulNBits the dynamic operator wins only at
  M=8-64 (W2/W8 and the larger shape); section 8.17.2 shows that prepacked
  MatMulNBits is faster still, so this window does not exist for production
  weights.
- At equal M the standard-layout fine kernel is 2-3x slower than the
  prepacked kernel (for example W4 M=128: 248 vs 123 us), because it decodes
  codes from a row-major layout with `DecodeWeight4` and has no fragment-ready
  layout; prepacking remains the right choice for block-32 serving weights.
- INT4 gains least from more row fragments because nibble decode and the
  promotion dominate; INT2 and INT8 gain most.
- The multi-fragment thresholds (96 CTAs) and the fine mode are tuned for
  108 SMs and have only been measured on A100. Code-generation side effects are
  large (change 3: a register attribute alone cost 7%), so any further kernel
  edit should be A/B-tested against the previous binary with interleaved runs.
- `mnb_pre` could not be measured in the build used above
  (`onnxruntime_USE_FPA_INTB_GEMM=OFF`), so `mnb_std` is only a proxy and
  overstates MatMulNBits' latency; see section 8.17.2 for the real baseline.

#### 8.17.2 Prepacked MatMulNBits baseline

The build was reconfigured with `onnxruntime_USE_FPA_INTB_GEMM=ON` (compact
tactic set, SM80 only) so `bench_prepacked.py` could time `mnb_pre`: the
MatMulNBits fpA_intB CUTLASS kernel on the same prepacked bytes. Reconfiguring
an existing build directory re-applies the FetchContent patches and fails with
"Reversed (or previously applied) patch" unless the sub-build `.ninja_log`
files are removed and their stamps touched; configure from the same path form
that created the directory, or every object is rebuilt. Idle A100-SXM4-80GB
(a first attempt on a GPU that another job started using showed ~340 us for
every case and was discarded), same protocol, microseconds, final code of
section 8.17.1:

| K x N | Weights | M | dyn_pre | mnb_pre | mnb_pre time / dyn_pre time (below 1: MatMulNBits faster) |
|-------|---------|---|---------|---------|------------------------------|
| 4096x4096 | W2 b64 | 1 | 40.1 | 20.6 | 0.51x |
| 4096x4096 | W2 b64 | 8 | 38.9 | 32.1 | 0.82x |
| 4096x4096 | W2 b64 | 16 | 40.6 | 35.7 | 0.88x |
| 4096x4096 | W2 b64 | 32 | 51.1 | 39.6 | 0.78x |
| 4096x4096 | W2 b64 | 64 | 60.6 | 44.9 | 0.74x |
| 4096x4096 | W2 b64 | 128 | 86.9 | 54.0 | 0.62x |
| 4096x4096 | W2 b64 | 256 | 128.8 | 70.9 | 0.55x |
| 4096x4096 | W2 b64 | 512 | 194.5 | 110.1 | 0.57x |
| 4096x4096 | W4 b32 | 1 | 50.9 | 21.9 | 0.43x |
| 4096x4096 | W4 b32 | 8 | 51.4 | 28.7 | 0.56x |
| 4096x4096 | W4 b32 | 16 | 51.1 | 36.4 | 0.71x |
| 4096x4096 | W4 b32 | 32 | 66.2 | 40.1 | 0.61x |
| 4096x4096 | W4 b32 | 64 | 84.4 | 46.2 | 0.55x |
| 4096x4096 | W4 b32 | 128 | 123.3 | 55.7 | 0.45x |
| 4096x4096 | W4 b32 | 256 | 186.5 | 73.5 | 0.39x |
| 4096x4096 | W4 b32 | 512 | 341.0 | 113.7 | 0.33x |
| 4096x4096 | W8 b32 | 1 | 49.3 | 20.9 | 0.42x |
| 4096x4096 | W8 b32 | 8 | 46.1 | 29.8 | 0.65x |
| 4096x4096 | W8 b32 | 16 | 49.8 | 37.0 | 0.74x |
| 4096x4096 | W8 b32 | 32 | 62.9 | 41.3 | 0.66x |
| 4096x4096 | W8 b32 | 64 | 81.9 | 47.4 | 0.58x |
| 4096x4096 | W8 b32 | 128 | 114.9 | 56.6 | 0.49x |
| 4096x4096 | W8 b32 | 256 | 171.3 | 77.5 | 0.45x |
| 4096x4096 | W8 b32 | 512 | 263.0 | 118.7 | 0.45x |
| 5120x6144 | W2 b64 | 1 | 47.2 | 22.4 | 0.47x |
| 5120x6144 | W2 b64 | 8 | 44.6 | 37.1 | 0.83x |
| 5120x6144 | W2 b64 | 16 | 47.3 | 39.5 | 0.83x |
| 5120x6144 | W2 b64 | 32 | 56.8 | 44.4 | 0.78x |
| 5120x6144 | W2 b64 | 64 | 76.3 | 50.2 | 0.66x |
| 5120x6144 | W2 b64 | 128 | 105.6 | 69.3 | 0.66x |
| 5120x6144 | W2 b64 | 256 | 190.0 | 101.5 | 0.53x |
| 5120x6144 | W2 b64 | 512 | 357.7 | 175.8 | 0.49x |
| 5120x6144 | W4 b32 | 1 | 60.7 | 24.2 | 0.40x |
| 5120x6144 | W4 b32 | 8 | 61.8 | 38.4 | 0.62x |
| 5120x6144 | W4 b32 | 16 | 61.8 | 41.0 | 0.66x |
| 5120x6144 | W4 b32 | 32 | 75.9 | 45.0 | 0.59x |
| 5120x6144 | W4 b32 | 64 | 103.9 | 52.1 | 0.50x |
| 5120x6144 | W4 b32 | 128 | 187.8 | 70.6 | 0.38x |
| 5120x6144 | W4 b32 | 256 | 350.1 | 105.2 | 0.30x |
| 5120x6144 | W4 b32 | 512 | 634.7 | 185.5 | 0.29x |
| 5120x6144 | W8 b32 | 1 | 65.9 | 33.2 | 0.50x |
| 5120x6144 | W8 b32 | 8 | 61.5 | 44.5 | 0.72x |
| 5120x6144 | W8 b32 | 16 | 66.7 | 42.8 | 0.64x |
| 5120x6144 | W8 b32 | 32 | 83.0 | 48.9 | 0.59x |
| 5120x6144 | W8 b32 | 64 | 104.4 | 55.8 | 0.53x |
| 5120x6144 | W8 b32 | 128 | 145.0 | 74.5 | 0.51x |
| 5120x6144 | W8 b32 | 256 | 263.4 | 109.5 | 0.42x |
| 5120x6144 | W8 b32 | 512 | 494.3 | 190.8 | 0.39x |

`Sm80PrepackedLayoutMatchesMatMulNBits` also passes in this build, and the
relative RMSE of the dynamic result against `mnb_pre` is 0.80-1.02% for every
case (activation quantization error).

Conclusion for A100: prepacked MatMulNBits is faster than the dynamic operator
at every measured M: 1.1-1.8x at M=8-16, 2.2-3.4x for W4 at M>=128 (W4, K=5120,
N=6144, M=512: 186 against 635 us), and it is more accurate. The
standard-layout MatMulNBits used as a baseline earlier was a poor proxy (for
example W4 K=N=4096, M=64: 63 us standard against 46 us prepacked). On A100 the
dynamic operator is therefore only useful where activation quantization itself
is wanted (for example to reproduce a W8A8 deployment), not as a speed-up over
weight-only fpA_intB. A speed-up would need a different promotion scheme than
bit-exact per-32-code FP32 accumulation.

Validation: `Sm80PrepackedMultiRowFragmentTiles`, `AmpereFineScaleIntervals`,
and the Python tests `test_sm80_prepacked_row_fragment_tiles_match_exact_reference`
and `test_row_counts_across_grouped_tile_selection_match_exact_reference`
(extended with block-32 and 96-wide activation geometries) compare bitwise with
the exact reference; an extended 1428-case sweep (28 row counts, 17 geometries,
all tile choices forced during development) also matched exactly.

### 8.18 RTX 4090 (SM89) dispatch and prepacked-kernel tuning

Measured on 2026-10-12 with an NVIDIA GeForce RTX 4090 (128 SMs, 24 GiB),
driver 610.60, Windows, Python 3.12.10, PyTorch 2.14.1+cu130, and ORT 1.32.0.
The build uses CUDA 13.3, cuDNN 9.24.1.1, `89-real`, Release, and the compact
INT2/INT4/INT8 FpA IntB kernel set. Flash Attention, memory-efficient attention,
and FP4/FP8 QMoE were disabled identically in both binaries. The CUDA 13.3
header's MSVC warning C4193 was suppressed with
`CMAKE_CUDA_FLAGS=-Xcompiler=/wd4193`; no source workaround was added.
Dynamic quantization retains `--ftz=false --prec-div=true --fmad=false`.

#### Summary

The baseline was commit `aec03d7e76e47ca27a52bd94cae5b041e9bb0bc3`.
Its SM89 dispatch used DP4A below M=32, the synchronous native fallback for
standard W4/W8 block-32 at larger M, and one prepacked row fragment at every M.
The final changes enable existing aligned/fine MMA kernels on SM89 from M=1
and reuse prepacked weight fragments across multiple row fragments.
Quantization, integer-dot semantics, scale boundaries, and FP32 accumulation
order are unchanged.

Measured replay medians include:

- Standard W4, K=5120/N=6144/M=16: **4011.01 -> 77.36 us (51.85x)**.
- Standard W8, K=N=4096/M=128: **529.56 -> 131.40 us (4.03x)**.
- Prepacked W8, K=5120/N=6144/M=512: **421.96 -> 227.91 us (1.85x)**.

These are measured medians, not guaranteed speedups. This live Windows
desktop has substantial timing variability, explicitly flagged below.

#### Metrics and methodology

The `bench_prepacked.py` was used in this test.
An artifact-only wrapper retained every trial, input/output hashes, loaded
DLL hashes, and CUDA provider settings. All inputs, scales, and outputs were
FP16; activation quantization was `dynamic_per_token`, with CUDA graphs and
GPU IOBinding on the current PyTorch CUDA stream. W2 uses block 64; W4/W8
use block 32.

For each case/label, a baseline pilot chose a repeat count targeting 20 ms
per trial. That plan was frozen for every before/after run. Each label had
50 warmups and 21 trials. Six complete, non-overlapping processes ran in
the order B1/A1/B2/A2/B3/A3, with no concurrent build or profiling.
Each table entry is the median of the three pass medians, in microseconds;
speedup is before/after. The event interval includes GPU idle gaps and
the cost of submitting/synchronizing ORT CUDA graph replays: it is not
isolated kernel time.

`std` is DynamicQuantizeMatMulNBits with standard blobs; `pre` is the same
operator with `weight_prepacked=1`. `control` is prepacked MatMulNBits on
the same weight bytes and scales, without activation quantization.
Control output bits varied across independent sessions, so only the dynamic
outputs are required to match the frozen baseline bit-for-bit. Control
outputs must remain finite, with a positive finite reference norm and finite
relative RMSE; the cause of the control bit variation was not established.

Quality `S/P` refers to standard/prepacked timing. `OK` requires all three
within-pass trial IQR/median ratios <= 5% and pass-median spread <= 3% in
both phases; `!` means at least one condition failed. Only two of the 96
dynamic timing comparisons meet this strict joint gate. The tables retain
every case instead of dropping noisy results. Control median drift exceeds
10% in three small-M cases, further limiting small-difference claims.

| K x N | W | M | std B -> A us | std x | pre B -> A us | pre x | control B -> A us | S/P |
|-------|---|---|---------------|-------|---------------|-------|-------------------|-----|
| 4096x4096 | 2 | 1 | 105.02 -> 51.46 | 2.04 | 34.20 -> 31.90 | 1.07 | 23.91 -> 23.69 | !/! |
| 4096x4096 | 2 | 8 | 593.30 -> 42.28 | 14.03 | 33.90 -> 33.60 | 1.01 | 30.62 -> 29.45 | !/! |
| 4096x4096 | 2 | 9 | 652.41 -> 44.55 | 14.65 | 39.43 -> 33.70 | 1.17 | 31.91 -> 29.74 | !/! |
| 4096x4096 | 2 | 16 | 1133.35 -> 50.23 | 22.56 | 34.03 -> 34.97 | 0.97 | 30.96 -> 31.21 | !/! |
| 4096x4096 | 2 | 32 | 83.55 -> 82.91 | 1.01 | 37.18 -> 33.91 | 1.10 | 33.00 -> 31.94 | !/! |
| 4096x4096 | 2 | 64 | 84.72 -> 85.63 | 0.99 | 40.91 -> 37.73 | 1.08 | 41.31 -> 38.00 | !/! |
| 4096x4096 | 2 | 128 | 100.69 -> 93.50 | 1.08 | 56.98 -> 44.96 | 1.27 | 53.99 -> 51.36 | !/! |
| 4096x4096 | 2 | 512 | 245.00 -> 243.57 | 1.01 | 156.75 -> 98.80 | 1.59 | 134.20 -> 133.04 | !/! |
| 4096x4096 | 4 | 1 | 180.13 -> 69.24 | 2.60 | 43.09 -> 43.11 | 1.00 | 25.34 -> 22.96 | !/! |
| 4096x4096 | 4 | 8 | 1128.78 -> 62.21 | 18.14 | 40.07 -> 39.85 | 1.01 | 29.36 -> 27.63 | !/! |
| 4096x4096 | 4 | 9 | 1279.99 -> 67.75 | 18.89 | 42.15 -> 38.36 | 1.10 | 30.70 -> 33.33 | !/! |
| 4096x4096 | 4 | 16 | 2155.02 -> 66.91 | 32.21 | 39.30 -> 40.64 | 0.97 | 31.79 -> 30.70 | !/! |
| 4096x4096 | 4 | 32 | 468.03 -> 71.71 | 6.53 | 40.43 -> 41.33 | 0.98 | 31.99 -> 32.30 | !/! |
| 4096x4096 | 4 | 64 | 487.19 -> 89.98 | 5.41 | 49.08 -> 48.27 | 1.02 | 37.93 -> 40.04 | !/! |
| 4096x4096 | 4 | 128 | 494.44 -> 131.07 | 3.77 | 79.11 -> 56.53 | 1.40 | 51.69 -> 53.15 | !/! |
| 4096x4096 | 4 | 512 | 1282.72 -> 390.05 | 3.29 | 236.70 -> 165.66 | 1.43 | 131.58 -> 133.44 | OK/! |
| 4096x4096 | 8 | 1 | 173.92 -> 63.29 | 2.75 | 38.83 -> 39.24 | 0.99 | 23.29 -> 24.99 | !/! |
| 4096x4096 | 8 | 8 | 1100.71 -> 61.05 | 18.03 | 37.34 -> 38.04 | 0.98 | 34.22 -> 29.70 | !/! |
| 4096x4096 | 8 | 9 | 1250.40 -> 59.24 | 21.11 | 38.64 -> 38.02 | 1.02 | 29.93 -> 30.00 | !/! |
| 4096x4096 | 8 | 16 | 2155.88 -> 63.12 | 34.16 | 38.43 -> 39.42 | 0.97 | 32.75 -> 32.63 | !/! |
| 4096x4096 | 8 | 32 | 507.55 -> 68.36 | 7.42 | 41.07 -> 39.09 | 1.05 | 35.43 -> 33.15 | !/! |
| 4096x4096 | 8 | 64 | 523.24 -> 89.37 | 5.85 | 47.36 -> 46.38 | 1.02 | 38.48 -> 40.27 | !/! |
| 4096x4096 | 8 | 128 | 529.56 -> 131.40 | 4.03 | 84.09 -> 54.29 | 1.55 | 54.97 -> 54.65 | !/! |
| 4096x4096 | 8 | 512 | 1280.84 -> 383.70 | 3.34 | 235.45 -> 127.94 | 1.84 | 134.21 -> 136.11 | !/! |
| 5120x6144 | 2 | 1 | 165.24 -> 61.82 | 2.67 | 36.67 -> 39.26 | 0.93 | 24.53 -> 27.19 | !/! |
| 5120x6144 | 2 | 8 | 1049.33 -> 58.12 | 18.06 | 37.66 -> 37.60 | 1.00 | 34.57 -> 34.10 | !/! |
| 5120x6144 | 2 | 9 | 1176.04 -> 62.10 | 18.94 | 41.23 -> 42.26 | 0.98 | 34.29 -> 34.99 | !/! |
| 5120x6144 | 2 | 16 | 2066.33 -> 60.99 | 33.88 | 36.62 -> 39.48 | 0.93 | 34.42 -> 35.83 | !/! |
| 5120x6144 | 2 | 32 | 95.52 -> 99.30 | 0.96 | 45.84 -> 45.40 | 1.01 | 40.37 -> 40.47 | !/! |
| 5120x6144 | 2 | 64 | 107.26 -> 111.53 | 0.96 | 52.51 -> 53.36 | 0.98 | 51.20 -> 52.12 | !/! |
| 5120x6144 | 2 | 128 | 124.98 -> 126.52 | 0.99 | 81.66 -> 69.43 | 1.18 | 75.00 -> 81.18 | !/! |
| 5120x6144 | 2 | 512 | 322.37 -> 325.42 | 0.99 | 266.61 -> 177.61 | 1.50 | 224.03 -> 229.89 | !/! |
| 5120x6144 | 4 | 1 | 303.76 -> 82.94 | 3.66 | 49.50 -> 46.14 | 1.07 | 26.88 -> 31.20 | !/! |
| 5120x6144 | 4 | 8 | 2049.77 -> 92.50 | 22.16 | 43.99 -> 44.28 | 0.99 | 33.25 -> 31.51 | !/! |
| 5120x6144 | 4 | 9 | 2295.03 -> 93.19 | 24.63 | 43.46 -> 44.50 | 0.98 | 35.86 -> 35.75 | !/! |
| 5120x6144 | 4 | 16 | 4011.01 -> 77.36 | 51.85 | 46.39 -> 46.80 | 0.99 | 32.86 -> 33.77 | !/! |
| 5120x6144 | 4 | 32 | 590.33 -> 101.27 | 5.83 | 61.59 -> 57.70 | 1.07 | 39.99 -> 40.31 | !/! |
| 5120x6144 | 4 | 64 | 665.04 -> 154.55 | 4.30 | 71.08 -> 68.53 | 1.04 | 53.48 -> 53.11 | OK/! |
| 5120x6144 | 4 | 128 | 749.69 -> 241.17 | 3.11 | 120.68 -> 95.16 | 1.27 | 80.39 -> 80.95 | !/! |
| 5120x6144 | 4 | 512 | 2382.30 -> 498.32 | 4.78 | 413.39 -> 290.97 | 1.42 | 228.79 -> 230.31 | !/! |
| 5120x6144 | 8 | 1 | 306.44 -> 81.15 | 3.78 | 43.70 -> 48.11 | 0.91 | 27.54 -> 26.72 | !/! |
| 5120x6144 | 8 | 8 | 2046.45 -> 75.03 | 27.28 | 41.82 -> 42.03 | 1.00 | 39.15 -> 37.10 | !/! |
| 5120x6144 | 8 | 9 | 2293.69 -> 82.51 | 27.80 | 44.02 -> 44.24 | 1.00 | 36.13 -> 33.84 | !/! |
| 5120x6144 | 8 | 16 | 3979.21 -> 80.47 | 49.45 | 44.11 -> 44.77 | 0.99 | 36.24 -> 35.74 | !/! |
| 5120x6144 | 8 | 32 | 639.11 -> 121.10 | 5.28 | 55.79 -> 56.32 | 0.99 | 44.53 -> 41.33 | !/! |
| 5120x6144 | 8 | 64 | 700.27 -> 155.40 | 4.51 | 74.94 -> 66.35 | 1.13 | 56.46 -> 59.79 | !/! |
| 5120x6144 | 8 | 128 | 761.18 -> 231.47 | 3.29 | 131.17 -> 82.51 | 1.59 | 85.09 -> 84.89 | !/! |
| 5120x6144 | 8 | 512 | 2339.49 -> 542.74 | 4.31 | 421.96 -> 227.91 | 1.85 | 240.04 -> 248.74 | !/! |

#### Findings

Nsight Systems 2026.3.2 verifies actual dispatch and isolates GPU kernel
duration. These diagnostic captures have 50 warmups and 20 traced replays;
they are separate from the unprofiled benchmark above.

| FP16 W4, K=N=4096 | Baseline kernel | Final kernel | Median GPU us B -> A |
|------------------|-----------------|--------------|---------------------|
| Standard M=16 | `IntegerMatMul` / DP4A | `NativeAlignedGroupedMatMulCapped`, fine intervals | 2094.930 -> 50.016 |
| Prepacked M=128 | `PrepackedMatMul`, one row fragment | `PrepackedMatMul`, two row fragments | 59.568 -> 39.216 |
| Quantizer, standard M=16 | `QuantizeRowsVector` | Same variant | 2.080 -> 2.208 |
| Quantizer, prepacked M=128 | `QuantizeRowsVector` | Same variant | 2.432 -> 2.400 |

Matmul accounts for 99.9% of baseline standard-M16 GPU kernel time and
95.7% after tuning. Thus this case is kernel/dispatch dominated, not
quantizer dominated. The matmul kernel improves 41.89x in this trace;
prepacked M128 improves 1.52x by sharing weight decoding across rows.
Nsight Compute 2026.3.0 could not access counters (`ERR_NVGPUCTRPERM`).
Compute/memory bandwidth, SOL, occupancy, and register-pressure bottlenecks
are therefore not claimed. Stream-synchronization waits overlap GPU work
and must not be added as independent host computation.

An initial candidate also enlarged coarse-interval tiles on SM89. For
standard W2, K=5120/N=6144/M=128, three paired passes showed
128.43 -> 159.64 us, a 24.3% slowdown. That candidate was discarded.
The final implementation retains the old coarse-interval selection at
M >= 32; the final 124.98 -> 126.52 us result is mixed/marginal, not a
claimed improvement. Other GPU architectures retain their previous rules.
Prepacked M <= 32 also keeps one row fragment; its small timing differences
are not claimed as kernel improvements.

Every final pass has 48 cases and 144 positive finite timing series.
All 576 dynamic output-hash comparisons match the frozen baseline, and
standard/prepacked dynamic outputs match bit-for-bit in every pass.
Relative RMSE against MatMulNBits is 0.806-1.024%. The final provider build
and full Python session suite passed: **19 tests and 590 subtests**,
including tile boundaries, all weight widths, all activation modes,
FP16/BF16, all scale types, and CUDA graph replay.

#### Reproduction and recommendations

The original worktree switched branches during the first build, so the
verified build is isolated at `D:\git\onnxruntime-dqmm-sm89`, based on the
baseline commit above. Its Python package is at
`build\sm89\Release\Release`. Measurement artifacts, wrapper, repeat plan,
preserved baseline/final DLLs, all raw trials, and profiler reports remain at
`D:\perf-artifacts\dqmm_session_20261012_134523`.
`paired-final-comparison.json` and `.md` contain the complete comparison
and quality flags; `profiling-evidence-readback.json` and
`profiler-full-kernel-names.json` retain profiler provenance and routing.

Baseline CUDA DLL SHA256:
`63c186c68f85fc4dfdb1c2de4ffbe7859df4a28f43340c52c887968436874974`.
Final CUDA DLL SHA256:
`bcb7de0f6f34d9ae192c41a754b00d6c5918984a33c63eb21ac1c400ed963ddc`.

With the build settings above, the incremental final build command was:

```powershell
& $env:ComSpec /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.51.36231 >nul && call D:\py312\Scripts\activate && "D:\py312\Lib\site-packages\cmake\data\bin\cmake.exe" --build D:\git\onnxruntime-dqmm-sm89\build\sm89\Release --config Release --target onnxruntime_providers_cuda --parallel 8 -- /p:BuildInParallel=true'
```

Result: passed; the edited `.cu` was recompiled and the CUDA DLL relinked.
After staging the DLL into the package's `onnxruntime\capi` directory,
the correctness command, run outside the source root, was:

```powershell
$env:PYTHONPATH = 'D:\git\onnxruntime-dqmm-sm89\build\sm89\Release\Release'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location 'D:\perf-artifacts\dqmm_session_20261012_134523'
& D:\py312\Scripts\python.exe -c "import torch,pytest; raise SystemExit(pytest.main(['-q',r'D:\git\onnxruntime-dqmm-sm89\onnxruntime\test\python\contrib_ops\test_dynamic_quantize_matmul_nbits.py']))"
```

Result: 19 passed, 590 subtests passed. Importing PyTorch first preloads
the CUDA DLLs, matching the supplied benchmark. A run without that preload
failed to load CUDA EP dependencies (Windows error 126), not a kernel
correctness assertion. The only passing-run warning was missing build
package metadata.

The six final benchmark processes used this command for each phase/pass:

```powershell
$bin = 'D:\git\onnxruntime-dqmm-sm89\build\sm89\Release\Release'
$artifacts = 'D:\perf-artifacts\dqmm_session_20261012_134523'
$env:PYTHONPATH = $bin
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;D:\cudnn\9.24.1.1_cuda13\bin\x64;' + $env:PATH
Set-Location $artifacts
foreach ($i in 1..3) {
  foreach ($phase in @('baseline', 'after')) {
    $folder = if ($phase -eq 'baseline') { 'baseline-binaries' } else { 'final-binaries' }
    $dll = Join-Path $artifacts "$folder\onnxruntime_providers_cuda.dll"
    Copy-Item $dll -Destination $bin -Force
    Copy-Item $dll -Destination (Join-Path $bin 'onnxruntime\capi') -Force
    & D:\py312\Scripts\python.exe .\benchmark_wrapper.py --phase $phase --pass-id "final-$i" --quality robust --ort-root D:\git\onnxruntime-dqmm-sm89 --expected-ort-root $bin --warmup 50 --trials 21 --adaptive --target-trial-ms 20 --repeat-plan .\full-repeat-plan-v2.json --out ".\$phase-final-$i.json"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $phase pass $i" }
  }
}
```

Result: all six passes complete. Choose unused output names to rerun;
the wrapper deliberately refuses to overwrite existing evidence.

Recommendations: keep the measured SM89 gates rather than extending them
to unmeasured architectures. Prepacking remains important, but W4 M=512
still loses to weight-only MatMulNBits; activation quantization is not a
universal speedup. Producer-side quantization fusion and deeper scale/
decode scheduling are future work. Recheck those changes with the exact
INT32-per-interval contract and quieter or clock-controlled measurements
before claiming small improvements. TensorRT-LLM's local Ada blockwise
FP8 kernels provide tiling/pipeline ideas, but are not a numerical drop-in
for this operator's INT2/INT4/INT8 common-scale intervals; no external
kernel source was copied.

### 8.19 RTX 4090 large-M follow-up and the #33200 use case (2026-10-12)

#### Summary

The section 8.18 sweep was extended to **M=1024 and 1536**, using the same
preserved before/after binaries and build configuration. Larger M changes
the conclusion: prepacked W2/block64 is faster than the prepacked
MatMulNBits control, W8/block32 has smaller and less stable advantages,
and W4/block32 still loses. Increasing M alone does not make activation
quantization a universal replacement for weight-only inference.

[Issue #33200](https://github.com/microsoft/onnxruntime/issues/33200)
requests a different regime: **per-token A8 and one W8 scale per output
channel across all K**, with one INT32 dot and a final scaling epilogue.
The original block32/block64 sweep does not test that premise. A separate
synthetic benchmark below exercises its four shapes and both larger M
values. It shows a useful but modest local niche, not a reproduction of
the author's other GPUs or claimed model-level latency/accuracy result.
No further kernel changes were made for this follow-up.

#### Metrics: extended blockwise before/after sweep

Protocol: B1/A1/B2/A2/B3/A3, 50 warmups, 21 trials, a new frozen adaptive
repeat plan targeting 20 ms per trial, and median of three pass medians.
The selection has 12 cases rather than 48. Inputs, weights, scales,
repeat counts, and provider settings match across the six processes.
Times below are **us per CUDA graph replay**, including activation
quantization for the dynamic operator and submission-related GPU gaps;
they are not isolated GEMM kernel times.

`B / A` means baseline / optimized. The first two `x` columns are B/A.
`MNB / Dyn` compares the optimized prepacked dynamic operator with the
MatMulNBits control measured in the optimized passes; >1 favors dynamic.
Quality S/P uses the same before/after gate as section 8.18: all three
trial IQRs <=5% and pass-median spread <=3% in both phases. `!` is flagged,
not a stability-certified improvement.

| KxN | W/block | M | Standard B / A us | x | Prepacked B / A us | x | MNB B / A us | MNB / Dyn | Quality S/P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4096x4096 | 2/64 | 1024 | 368.58 / 370.15 | 1.00 | 277.99 / 174.87 | 1.59 | 231.07 / 231.26 | 1.32 | !/! |
| 4096x4096 | 2/64 | 1536 | 511.19 / 513.51 | 1.00 | 412.54 / 255.20 | 1.62 | 337.94 / 339.73 | 1.33 | OK/OK |
| 4096x4096 | 4/32 | 1024 | 2552.13 / 571.30 | 4.47 | 441.99 / 306.89 | 1.44 | 235.55 / 238.57 | 0.78 | !/OK |
| 4096x4096 | 4/32 | 1536 | 3798.14 / 761.72 | 4.99 | 652.79 / 454.70 | 1.44 | 344.23 / 343.94 | 0.76 | !/OK |
| 4096x4096 | 8/32 | 1024 | 2509.79 / 566.51 | 4.43 | 434.72 / 228.47 | 1.90 | 243.33 / 243.02 | 1.06 | !/! |
| 4096x4096 | 8/32 | 1536 | 3736.75 / 763.58 | 4.89 | 638.28 / 330.87 | 1.93 | 357.83 / 356.03 | 1.08 | !/! |
| 5120x6144 | 2/64 | 1024 | 620.87 / 621.99 | 1.00 | 512.90 / 321.41 | 1.60 | 419.50 / 415.31 | 1.29 | OK/! |
| 5120x6144 | 2/64 | 1536 | 923.09 / 924.08 | 1.00 | 761.08 / 473.61 | 1.61 | 626.32 / 616.10 | 1.30 | OK/OK |
| 5120x6144 | 4/32 | 1024 | 4843.33 / 936.77 | 5.17 | 813.31 / 573.23 | 1.42 | 426.28 / 433.87 | 0.76 | !/OK |
| 5120x6144 | 4/32 | 1536 | 7247.10 / 1385.47 | 5.23 | 1219.64 / 848.60 | 1.44 | 635.05 / 635.81 | 0.75 | !/OK |
| 5120x6144 | 8/32 | 1024 | 4730.88 / 963.63 | 4.91 | 823.51 / 422.04 | 1.95 | 444.85 / 446.54 | 1.06 | !/! |
| 5120x6144 | 8/32 | 1536 | 7112.69 / 1407.16 | 5.05 | 1219.56 / 629.42 | 1.94 | 725.38 / 726.93 | 1.15 | !/! |

All six passes completed: 216 timing series, **4,536 raw trials**, and
**144 strict dynamic output-hash matches** to the frozen baseline.
Standard/prepacked dynamic outputs remain bit-identical in every case.
Relative RMSE versus MatMulNBits is **0.859-0.884%** on these inputs.
Nine of 24 dynamic before/after comparisons meet the combined quality
gate. Seven of 12 optimized prepacked/control comparisons meet their
post-change gate; no control's aggregate before/after drift exceeds 10%.

Prepacked W2 wins in all three optimized passes at every shape/M, with
aggregate speedups of **1.29-1.33x** over MatMulNBits. W4 is approximately
29-33% slower despite the tuning. W8's aggregate **1.06-1.15x** advantages
need caution: all four comparisons are quality-flagged, and
4096x4096/M1024 actually loses to the control in one of the three passes.
The standard W2 coarse-interval dispatch is unchanged; its approximately
1.00x before/after ratios are not optimization claims.

#### Metrics: per-channel W8A8 matching the issue's arithmetic regime

This independent sweep uses FP16 input/output, per-token activation
scales, and standard W8 codes with **one FP16 weight scale per channel**.
The dynamic block size is 2048 for K2048 and 8192 for K6144; the latter
pads the last 2048 codes per channel with midpoint code128. Only the
logical K6144 entries contribute to the dot product.

The same logical codes and channel-scale values supply all three
controls; weights are not independently requantized:

- **MNB b128:** standard-layout W8/block128, replicating the channel scale
  over K blocks. This matches the issue's reported comparison format,
  but this local compact build does not include the W8/block128
  fpA_intB specialization, so it uses the dequantize/GEMM fallback.
- **Pre MNB b32:** SM80-prepacked W8/block32, again repeating the same
  channel scale. This exercises the available compact mixed-GEMM path
  rather than relying solely on the block128 fallback. It is not always
  faster than that fallback at these shapes.
- **Dense:** ordinary FP16 MatMul with the weights dequantized once to
  FP16 outside timing. This is a practical latency control, but uses
  larger persistent weight storage than either quantized operator.

Three independent passes use 50 warmups and 21 trials with their own
frozen adaptive repeat plan. Timing-label order rotates across passes.
Values are medians of three pass medians, including dynamic quantization.
Each `x` is control/dynamic. Quality b128/pre/dense requires both compared
labels to satisfy all IQR <=5% and pass spread <=3% checks.

| KxN | M | Dynamic us | MNB b128 us | x | Pre MNB b32 us | x | Dense us | x | RMSE vs b128 | Quality b128/pre/dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2048x6144 | 1024 | 166.94 | 189.94 | 1.138 | 188.91 | 1.132 | 167.74 | 1.005 | 0.008293 | !/!/! |
| 2048x6144 | 1536 | 242.94 | 275.75 | 1.135 | 306.82 | 1.263 | 258.95 | 1.066 | 0.008301 | !/!/! |
| 6144x2048 | 1024 | 164.29 | 194.90 | 1.186 | 191.14 | 1.163 | 181.74 | 1.106 | 0.008869 | !/!/! |
| 6144x2048 | 1536 | 230.91 | 276.47 | 1.197 | 354.19 | 1.534 | 265.74 | 1.151 | 0.008906 | !/!/! |
| 2048x4096 | 1024 | 116.11 | 130.29 | 1.122 | 130.08 | 1.120 | 120.79 | 1.040 | 0.008275 | OK/OK/OK |
| 2048x4096 | 1536 | 163.63 | 185.14 | 1.131 | 200.57 | 1.226 | 174.28 | 1.065 | 0.008237 | !/!/! |
| 2048x2048 | 1024 | 75.54 | 71.13 | 0.942 | 71.73 | 0.950 | 66.17 | 0.876 | 0.008276 | !/!/! |
| 2048x2048 | 1536 | 95.53 | 106.87 | 1.119 | 141.80 | 1.484 | 100.38 | 1.051 | 0.008257 | !/!/! |

All 96 timing series and **2,016 raw trials** are positive and finite.
All **24 full dynamic-output hashes** match the independent frozen plan.
For each case/pass, 120 sampled FP16 outputs are bit-exact against a
CPU INT64 full-K dot, independently computed NumPy FP32 activation
quantization, and separate ordered FP32 scaling/conversion: **2,880
reference elements checked**. This sample check is not a claim of
full-output equality to the independent reference. The full outputs
are finite; dynamic relative RMSE versus b128 is **0.824-0.891%**.
No application-accuracy acceptance threshold was supplied or tested.

#### Findings

1. **The use case is credible, but narrower than a universal speedup.**
   For the first three issue shapes, both M values win over b128 in all
   three passes by more than 3%, with aggregate **1.12-1.20x** speedups.
   Only 2048x4096/M1024 passes the full stability gate. The 2048x2048
   results are mixed: M1024 loses in the aggregate, and M1536's aggregate
   gain over b128 is not consistent across passes. Wins over prepacked
   b32 reach 1.53x, but b128 or dense FP16 can be faster controls; do not
   select the slowest control to claim a general deployment benefit.
2. **Scale granularity matters as much as M.** A block32 W8 dot must
   promote/scale INT32 partial sums every 32 K elements, versus one
   promotion/epilogue for per-channel W8. The issue's integer-throughput
   argument is therefore not contradicted by a slower fine-block
   benchmark. Conversely, its one-scale result does not promise a gain
   for blockwise W4. The exact interval-order contract forbids silently
   merging equal adjacent scales to imitate the per-channel path.
3. **These are operator-level replay timings, not a model result.**
   The issue author's RTX4080 Laptop/GB10 measurements and approximately
   25% classifier latency reduction were not reproduced. No counters
   were collected for these new cases, and no compute-/memory-bound,
   occupancy, bandwidth, or SOL conclusion is inferred from latency.
   The earlier section 8.18 kernel traces remain evidence for the
   previously profiled cases only.
4. **W8A8 is an accuracy choice.** MatMulNBits retains FP16 activations;
   this operator deliberately quantizes them. Synthetic RMSE and exact
   implementation-reference checks do not establish unchanged task
   accuracy. Per-token quantization keeps tokens independent, but does
   not remove within-token feature outliers.

#### Recommendations and reproduction

Use this operator as a **per-node opt-in** for models validated for A8,
especially large-M prefill/encoder W8 with per-channel weights, or the
measured prepacked W2/block64 cases. Its purpose is native GPU dynamic
activation quantization plus integer GEMM and fused output scaling, not
additional persistent weight compression over MatMulNBits. Inputs and
outputs remain FP16 here, and quantized activations are internal scratch,
not an automatic reduction of the model's persistent activation storage.

Keep weight-only MatMulNBits for the measured W4/block32 cases, small
shapes without a demonstrated gain, and accuracy-sensitive/outlier-heavy
nodes unless smoothing or another activation policy is model-validated.
Select against the fastest applicable weight-only control, measure task
accuracy and complete-model latency, and rerun modest gains under quieter
GPU conditions before enabling them by default. Future per-channel
pipeline/tile tuning is separate work; larger-M results alone do not
justify changing another architecture's dispatch.

The original artifact directory from section 8.18 retains the evidence:
`large-m-repeat-plan-v2.json`, `large-m-baseline-[1..3].json`,
`large-m-after-[1..3].json`, and `large-m-final-comparison.json` / `.md`.
The corresponding per-channel files are
`issue33200-channel-dense-repeat-plan.json`,
`issue33200-channel-dense-[1..3].json`, and
`issue33200-channel-dense-summary.json` / `.md`.
The accepted final provider remains installed in both isolated build
locations; its SHA256 is unchanged from section 8.18.

For the larger-M extension, use the section 8.18 DLL-swap loop and
environment with these arguments for each phase/pass:

```powershell
& D:\py312\Scripts\python.exe .\benchmark_wrapper.py --phase $phase --pass-id "large-m-v2-$i" --quality robust --ort-root D:\git\onnxruntime --expected-ort-root $bin --shapes 4096x4096,5120x6144 --m 1024 1536 --warmup 50 --trials 21 --adaptive --target-trial-ms 20 --repeat-plan .\large-m-repeat-plan-v2.json --out ".\large-m-$phase-$i.json"
```

Result: all six passes completed. On Windows PowerShell 5, redirect
stdout/stderr through `cmd.exe` when recording logs, rather than treating
the known package-metadata warning as a terminating native-command error.
Use fresh output/plan names when rerunning; completed evidence is not
overwritten. The CPU-only aggregation command was:

```powershell
& D:\py312\Scripts\python.exe .\compare_paired_large_m.py --repeat-plan large-m-repeat-plan-v2.json --baseline-prefix large-m-baseline --after-prefix large-m-after --candidate final --expected-after-provider-sha bcb7de0f6f34d9ae192c41a754b00d6c5918984a33c63eb21ac1c400ed963ddc --output-stem large-m-final-comparison
```

Result: passed all raw-trial, numerical, binary-identity, and interleaving
checks. After restoring the final DLL, the separate issue-regime command
was:

```powershell
& D:\perf-artifacts\dqmm_session_20261012_134523\run_issue33200.ps1 -IncludeDense -MaxSecondsPerPass 600
```

Result: the independent CPU reference self-check, all three GPU passes,
and aggregation passed. The artifact-only launcher uses
`bench_issue33200_per_channel.py` and `summarize_issue33200.py`; it fixes
the four issue shapes, checks the final provider identity, and records
the full commands, inputs/scales/weights hashes, raw trials, correctness
checks, and quality flags.

### 8.20 RTX 5060 Ti / RTX 3060 compact-GPU tuning

Measured on 2026-10-12 against the pre-tuning branch kernel at commit
`fdbddaf93c` (`perf(cuda): RTX 4090 tuning`). This section concerns only
`DynamicQuantizeMatMulNBits`; no MoE kernels or MoE documentation are part
of this tuning change.

#### Environment and accepted changes

| Item | Configuration |
|------|---------------|
| GPU 0 | RTX 5060 Ti, SM120, 36 SMs, 16 GB, WDDM |
| GPU 1 | RTX 3060, SM86, 28 SMs, 12 GB, WDDM; also drives the desktop |
| NVIDIA driver | 610.88 |
| Build | Bundled CUDA EP, Release, VS2022, CUDA 13.0.88, cuDNN 9.13.0.50 |
| CUDA targets | `86-real;120-real`; the build normalizes SM120 to `120a-real` |
| Python/runtime | Python 3.12, ORT 1.32.0 freshly built from this branch, Torch 2.10.0+cu130 |
| Other dependencies | NumPy 2.3.5, ONNX 1.22.0, ml_dtypes 0.6.0, onnx_ir 1.0.0 |
| Build controls | Flash Attention OFF in both builds; dynamic-quantization flags `--ftz=false --prec-div=true --fmad=false` retained |

The baseline is a fresh build, not the previously installed CPU-only ORT
package or an older CUDA DLL. Two frozen Python-runtime directories share
the same core runtime and differ only in the rebuilt CUDA provider DLL:

| Provider | SHA256 of `onnxruntime_providers_cuda.dll` |
|----------|------------------------------------------|
| Before | `83335a36ada82348ad65342bfbd0362981ec4a14c8abd4df09e996b66f1d36f7` |
| After | `0e0c5a60ef8f087fd7cb22431ba5601407ac21189e1123b079b207ebc2b3c120` |

Accepted changes:

1. Pair the two prepacked MMA B-register decodes. W2/W4 share two loaded
   words and byte permutations rather than decode each register separately.
   The byte-wise centering operation still prevents inter-byte borrows.
   W8 keeps its original byte-permutation mapping.
2. Gate new standard-layout dispatch and occupancy targets on SM86 or
   SM120 with at most 48 SMs. Use the aligned asynchronous pipeline from
   M=1 and enable the existing 32-aligned fine-interval path on these
   compact devices. Compact SM86 retains its established M >= 32
   `32 x 64` / `64 x 64` selection; compact SM120 selects among the four
   existing tile shapes using its actual SM count and padded-row cost.
3. Reuse prepacked weights across more row fragments when sufficient CTAs
   remain. Compact devices target `floor(4 * SM_count / 3)` CTAs: 48 on
   RTX 5060 Ti and 37 on RTX 3060. W2/W8 allow four fragments; W4 allows
   four on compact SM120 and two on compact SM86. M <= 16 retains one
   fragment, including the existing M=8/M=9 specializations.

The activation quantizer, floating scale arithmetic, common-interval
boundaries, accumulation order, bias semantics, and serialized weight
layout are unchanged. The single-interval W8 path and unsupported-geometry
fallbacks are preserved. Larger devices retain their established dispatch;
paired prepacked decoding is shared by SM80+ kernels, but its performance
was measured here only on the two listed GPUs. An experimental unused-row
pruning change did not give a reliable benefit and was not retained.

#### Methodology and timing quality

The checked wrapper imports the supplied
`D:\git\dev\scripts\dynamic_quantize_matmul_nbit\bench_prepacked.py` for
model/session creation and the source `prepack_sm80` helper. Each GPU has
48 cases: K x N = 4096 x 4096 and 5120 x 6144, M =
1/8/9/16/32/64/128/512, and W2/W4/W8. W2 uses block-64; W4/W8 use block-32.
Activations, weight scales, and outputs are FP16. Both dynamic layouts
use `dynamic_per_token`, with no bias in the timing sweep. `MatMulNBits`
uses the same prepacked weights as an unchanged weight-only control.

All sessions use CUDA graphs and GPU IOBinding on the current Torch CUDA
stream, with CPU EP fallback explicitly disabled. Each label receives at
least 50 warmups and at least 250 ms of warming. There are 15 trials per
label per pass. Baseline pass 1 chooses repeats for approximately 20 ms
per trial, bounded to 5..1000; the same per-case/per-label repeat counts
are frozen for all later baseline and optimized passes. The optimized
fast paths can therefore have substantially shorter trials than 20 ms.

Three baseline and three optimized passes were collected per GPU, with
each pass ordered baseline then optimized and GPUs run sequentially.
The table reports the **median of the three pass medians**, in
microseconds per invocation. These CUDA-event intervals include Python/ORT
submission and scheduling gaps between graph replays; they are not isolated
kernel times. Synchronized host-wall medians and all raw trial times are
retained in the JSON artifacts.

Every pass verifies finite outputs, identical standard/prepacked FP16
output bits, frozen input/weight/scale hashes, and exact dynamic output
hashes against baseline pass 1. The third RTX 3060 baseline process stalled
during shutdown after saving its complete result. Only that benchmark
session was terminated; all 48 cases, all 15 trials per label, frozen
repeat counts, and hashes in its saved JSON were independently verified
before inclusion. The two third optimized passes then completed normally.

`B` = before, `A` = after, `Gain` = B/A. `Q S/P` gives separate quality
flags for standard/prepacked dynamic timing. `OK` requires **every**
within-pass IQR/median <= 5% and pass-median spread
`(max - min) / median` <= 3% in **both** phases; `!` means at least one
threshold failed, not a numerical failure. No cases were discarded.

| GPU | Standard geomean | Prepacked geomean | Standard OK | Prepacked OK | Control geomean B/A | Control OK |
|-----|------------------|-------------------|-------------|--------------|--------------------|------------|
| RTX 5060 Ti | 7.04x | 1.29x | 38/48 | 35/48 | 1.006x | 27/48 |
| RTX 3060 | 10.87x | 1.37x | 26/48 | 16/48 | 1.006x | 18/48 |

The quality-qualified subsets have standard/prepacked geomeans of
7.38x/1.34x on RTX 5060 Ti and 8.76x/1.62x on RTX 3060. They contain
different subsets and are not a like-for-like comparison between GPUs.

#### RTX 5060 Ti, K=4096, N=4096

| W | M | Std B | Std A | Gain | Pre B | Pre A | Gain | MNB B / A | Q S/P |
|---|---|-------|-------|------|-------|-------|------|-----------|-------|
| 2 | 1 | 257.38 | 41.48 | 6.20x | 31.11 | 28.80 | 1.08x | 23.19 / 23.29 | OK/! |
| 2 | 8 | 58.41 | 42.66 | 1.37x | 32.00 | 28.28 | 1.13x | 32.46 / 32.30 | OK/! |
| 2 | 9 | 58.59 | 43.01 | 1.36x | 32.21 | 29.66 | 1.09x | 31.55 / 30.95 | !/! |
| 2 | 16 | 60.00 | 44.43 | 1.35x | 32.18 | 29.92 | 1.08x | 31.66 / 32.06 | OK/! |
| 2 | 32 | 60.32 | 55.66 | 1.08x | 46.15 | 40.30 | 1.15x | 43.70 / 43.83 | !/! |
| 2 | 64 | 81.89 | 81.29 | 1.01x | 72.46 | 49.57 | 1.46x | 68.69 / 68.21 | OK/! |
| 2 | 128 | 156.88 | 156.11 | 1.00x | 119.92 | 80.72 | 1.49x | 115.09 / 113.85 | OK/OK |
| 2 | 512 | 426.51 | 425.60 | 1.00x | 407.32 | 250.49 | 1.63x | 379.88 / 382.41 | OK/OK |
| 4 | 1 | 494.62 | 58.17 | 8.50x | 39.50 | 35.41 | 1.12x | 23.85 / 24.24 | OK/OK |
| 4 | 8 | 3497.88 | 61.35 | 57.01x | 38.81 | 35.14 | 1.10x | 34.41 / 33.62 | !/OK |
| 4 | 9 | 3894.11 | 60.37 | 64.50x | 39.88 | 36.62 | 1.09x | 31.65 / 31.92 | OK/! |
| 4 | 16 | 6926.60 | 62.07 | 111.60x | 39.83 | 36.22 | 1.10x | 31.95 / 31.79 | OK/OK |
| 4 | 32 | 581.81 | 80.59 | 7.22x | 59.70 | 43.29 | 1.38x | 44.27 / 43.98 | OK/OK |
| 4 | 64 | 622.80 | 123.35 | 5.05x | 100.08 | 60.50 | 1.65x | 72.78 / 71.46 | OK/OK |
| 4 | 128 | 1163.99 | 228.79 | 5.09x | 170.35 | 106.24 | 1.60x | 117.35 / 116.38 | OK/OK |
| 4 | 512 | 3191.52 | 641.30 | 4.98x | 600.46 | 352.24 | 1.70x | 388.32 / 388.49 | OK/OK |
| 8 | 1 | 490.89 | 55.58 | 8.83x | 41.59 | 40.70 | 1.02x | 24.77 / 24.87 | !/OK |
| 8 | 8 | 3479.89 | 58.96 | 59.02x | 39.80 | 39.03 | 1.02x | 33.57 / 32.80 | OK/OK |
| 8 | 9 | 3914.36 | 59.82 | 65.44x | 41.83 | 41.22 | 1.01x | 33.55 / 32.80 | OK/! |
| 8 | 16 | 6903.74 | 61.47 | 112.32x | 42.00 | 40.88 | 1.03x | 33.55 / 32.88 | OK/OK |
| 8 | 32 | 561.31 | 79.34 | 7.07x | 64.54 | 47.43 | 1.36x | 45.85 / 45.38 | OK/OK |
| 8 | 64 | 592.53 | 118.95 | 4.98x | 108.51 | 63.28 | 1.71x | 73.59 / 73.72 | OK/OK |
| 8 | 128 | 1116.78 | 225.18 | 4.96x | 188.43 | 111.56 | 1.69x | 120.59 / 119.82 | OK/OK |
| 8 | 512 | 3044.01 | 608.32 | 5.00x | 666.35 | 371.50 | 1.79x | 418.40 / 417.80 | OK/OK |

#### RTX 5060 Ti, K=5120, N=6144

| W | M | Std B | Std A | Gain | Pre B | Pre A | Gain | MNB B / A | Q S/P |
|---|---|-------|-------|------|-------|-------|------|-----------|-------|
| 2 | 1 | 452.24 | 72.47 | 6.24x | 43.68 | 39.24 | 1.11x | 25.64 / 25.53 | OK/! |
| 2 | 8 | 82.89 | 72.51 | 1.14x | 42.47 | 38.89 | 1.09x | 47.19 / 46.68 | !/OK |
| 2 | 9 | 83.23 | 72.52 | 1.15x | 45.78 | 41.13 | 1.11x | 42.74 / 43.77 | OK/! |
| 2 | 16 | 83.22 | 57.18 | 1.46x | 46.10 | 41.19 | 1.12x | 43.16 / 43.30 | !/! |
| 2 | 32 | 85.39 | 84.93 | 1.01x | 71.42 | 54.26 | 1.32x | 64.76 / 63.48 | OK/! |
| 2 | 64 | 151.42 | 119.42 | 1.27x | 114.63 | 76.18 | 1.50x | 106.31 / 105.42 | !/! |
| 2 | 128 | 223.08 | 221.98 | 1.00x | 208.12 | 138.28 | 1.51x | 190.36 / 189.69 | OK/OK |
| 2 | 512 | 784.58 | 791.92 | 0.99x | 767.37 | 456.06 | 1.68x | 700.85 / 697.06 | OK/OK |
| 4 | 1 | 881.34 | 106.36 | 8.29x | 58.38 | 51.26 | 1.14x | 26.54 / 26.12 | !/OK |
| 4 | 8 | 6524.99 | 110.87 | 58.85x | 55.69 | 51.37 | 1.08x | 48.61 / 47.74 | OK/OK |
| 4 | 9 | 7304.69 | 111.59 | 65.46x | 58.39 | 51.46 | 1.13x | 44.69 / 44.96 | OK/OK |
| 4 | 16 | 12933.38 | 88.08 | 146.83x | 59.05 | 51.33 | 1.15x | 44.90 / 44.93 | OK/OK |
| 4 | 32 | 743.96 | 121.50 | 6.12x | 96.97 | 65.80 | 1.47x | 63.76 / 63.90 | OK/OK |
| 4 | 64 | 862.95 | 173.60 | 4.97x | 162.31 | 107.29 | 1.51x | 108.78 / 108.32 | OK/OK |
| 4 | 128 | 1605.73 | 328.63 | 4.89x | 302.13 | 190.07 | 1.59x | 195.46 / 194.89 | OK/OK |
| 4 | 512 | 5977.04 | 1172.38 | 5.10x | 1125.01 | 639.91 | 1.76x | 706.98 / 707.41 | OK/OK |
| 8 | 1 | 882.75 | 105.91 | 8.33x | 62.42 | 61.80 | 1.01x | 35.07 / 33.28 | OK/OK |
| 8 | 8 | 6522.72 | 112.44 | 58.01x | 62.15 | 61.53 | 1.01x | 45.89 / 45.50 | !/OK |
| 8 | 9 | 7308.69 | 111.88 | 65.32x | 62.30 | 61.77 | 1.01x | 47.33 / 46.86 | !/OK |
| 8 | 16 | 12958.60 | 96.01 | 134.98x | 64.30 | 63.95 | 1.01x | 47.76 / 47.43 | OK/OK |
| 8 | 32 | 909.01 | 126.88 | 7.16x | 108.04 | 74.43 | 1.45x | 69.99 / 68.65 | OK/OK |
| 8 | 64 | 953.92 | 176.26 | 5.41x | 180.22 | 117.83 | 1.53x | 119.95 / 119.00 | OK/OK |
| 8 | 128 | 1830.95 | 326.13 | 5.61x | 336.72 | 205.98 | 1.63x | 214.00 / 213.26 | OK/OK |
| 8 | 512 | 6187.07 | 1181.34 | 5.24x | 1278.22 | 711.63 | 1.80x | 771.60 / 765.60 | OK/OK |

#### RTX 3060, K=4096, N=4096

| W | M | Std B | Std A | Gain | Pre B | Pre A | Gain | MNB B / A | Q S/P |
|---|---|-------|-------|------|-------|-------|------|-----------|-------|
| 2 | 1 | 481.46 | 79.13 | 6.08x | 54.44 | 51.33 | 1.06x | 33.37 / 32.82 | OK/! |
| 2 | 8 | 3457.83 | 81.37 | 42.50x | 52.59 | 48.53 | 1.08x | 48.54 / 48.60 | OK/! |
| 2 | 9 | 3867.96 | 84.44 | 45.81x | 53.77 | 51.46 | 1.04x | 47.62 / 47.95 | OK/! |
| 2 | 16 | 6811.79 | 86.07 | 79.15x | 55.43 | 52.35 | 1.06x | 48.56 / 48.64 | OK/! |
| 2 | 32 | 133.16 | 133.62 | 1.00x | 78.26 | 61.42 | 1.27x | 70.98 / 69.72 | !/OK |
| 2 | 64 | 222.31 | 221.66 | 1.00x | 132.08 | 96.44 | 1.37x | 114.92 / 112.53 | !/! |
| 2 | 128 | 358.42 | 354.65 | 1.01x | 233.72 | 147.06 | 1.59x | 205.30 / 200.23 | !/! |
| 2 | 512 | 1106.50 | 1103.10 | 1.00x | 854.80 | 495.97 | 1.72x | 695.64 / 696.21 | OK/! |
| 4 | 1 | 933.90 | 109.42 | 8.53x | 73.12 | 63.46 | 1.15x | 46.33 / 46.20 | !/! |
| 4 | 8 | 6779.23 | 110.53 | 61.33x | 70.25 | 61.15 | 1.15x | 52.51 / 52.44 | !/! |
| 4 | 9 | 7559.60 | 119.56 | 63.23x | 78.95 | 63.91 | 1.24x | 52.93 / 52.40 | OK/! |
| 4 | 16 | 13530.32 | 117.97 | 114.69x | 75.55 | 64.63 | 1.17x | 53.24 / 53.34 | !/! |
| 4 | 32 | 786.47 | 184.25 | 4.27x | 111.90 | 78.91 | 1.42x | 74.56 / 73.94 | OK/! |
| 4 | 64 | 1610.80 | 315.42 | 5.11x | 201.36 | 122.14 | 1.65x | 118.50 / 117.99 | !/! |
| 4 | 128 | 2553.07 | 505.31 | 5.05x | 362.77 | 221.31 | 1.64x | 210.22 / 208.20 | !/! |
| 4 | 512 | 8524.29 | 1657.93 | 5.14x | 1366.81 | 774.44 | 1.76x | 708.78 / 712.32 | !/! |
| 8 | 1 | 954.68 | 111.80 | 8.54x | 85.32 | 84.73 | 1.01x | 73.24 / 73.07 | !/! |
| 8 | 8 | 6893.31 | 114.89 | 60.00x | 85.05 | 84.02 | 1.01x | 77.05 / 77.61 | !/OK |
| 8 | 9 | 7729.71 | 117.08 | 66.02x | 85.39 | 86.15 | 0.99x | 80.04 / 78.45 | !/! |
| 8 | 16 | 13724.24 | 118.11 | 116.20x | 86.28 | 86.61 | 1.00x | 80.26 / 79.18 | !/! |
| 8 | 32 | 902.14 | 186.15 | 4.85x | 128.54 | 101.31 | 1.27x | 88.29 / 88.08 | OK/! |
| 8 | 64 | 1791.23 | 333.50 | 5.37x | 235.42 | 121.73 | 1.93x | 129.68 / 129.29 | OK/! |
| 8 | 128 | 2750.52 | 512.18 | 5.37x | 452.22 | 191.75 | 2.36x | 230.40 / 227.34 | OK/OK |
| 8 | 512 | 9297.17 | 1653.72 | 5.62x | 1779.41 | 670.28 | 2.65x | 785.49 / 781.51 | OK/OK |

#### RTX 3060, K=5120, N=6144

| W | M | Std B | Std A | Gain | Pre B | Pre A | Gain | MNB B / A | Q S/P |
|---|---|-------|-------|------|-------|-------|------|-----------|-------|
| 2 | 1 | 866.62 | 141.28 | 6.13x | 78.61 | 74.73 | 1.05x | 48.82 / 49.09 | !/! |
| 2 | 8 | 6464.35 | 142.97 | 45.21x | 76.37 | 71.51 | 1.07x | 69.72 / 69.44 | !/OK |
| 2 | 9 | 7237.53 | 150.33 | 48.14x | 78.62 | 75.47 | 1.04x | 69.89 / 69.38 | OK/! |
| 2 | 16 | 12972.39 | 125.56 | 103.32x | 80.01 | 76.04 | 1.05x | 70.30 / 68.94 | !/! |
| 2 | 32 | 185.01 | 186.06 | 0.99x | 121.64 | 95.41 | 1.27x | 107.58 / 107.80 | OK/! |
| 2 | 64 | 314.18 | 314.81 | 1.00x | 219.29 | 145.32 | 1.51x | 188.21 / 186.90 | !/! |
| 2 | 128 | 591.04 | 590.69 | 1.00x | 416.24 | 248.30 | 1.68x | 347.80 / 346.62 | OK/! |
| 2 | 512 | 1967.20 | 1965.44 | 1.00x | 1582.51 | 914.39 | 1.73x | 1287.44 / 1296.87 | OK/OK |
| 4 | 1 | 1697.86 | 209.20 | 8.12x | 116.42 | 99.89 | 1.17x | 71.85 / 72.19 | OK/OK |
| 4 | 8 | 12805.13 | 209.42 | 61.15x | 111.21 | 96.36 | 1.15x | 80.19 / 78.41 | !/! |
| 4 | 9 | 14337.38 | 224.47 | 63.87x | 119.10 | 101.19 | 1.18x | 78.39 / 78.37 | OK/! |
| 4 | 16 | 25441.21 | 176.38 | 144.24x | 121.22 | 102.03 | 1.19x | 80.19 / 79.22 | OK/! |
| 4 | 32 | 1119.21 | 260.58 | 4.30x | 186.92 | 126.56 | 1.48x | 114.59 / 114.60 | OK/OK |
| 4 | 64 | 2222.14 | 477.43 | 4.65x | 343.02 | 205.73 | 1.67x | 193.68 / 192.70 | OK/OK |
| 4 | 128 | 4401.80 | 900.58 | 4.89x | 663.71 | 386.29 | 1.72x | 353.46 / 351.44 | !/OK |
| 4 | 512 | 15672.97 | 3016.12 | 5.20x | 2541.89 | 1448.32 | 1.76x | 1304.26 / 1297.98 | OK/OK |
| 8 | 1 | 1719.11 | 216.49 | 7.94x | 155.51 | 154.66 | 1.01x | 119.68 / 119.10 | !/OK |
| 8 | 8 | 12946.53 | 221.43 | 58.47x | 155.09 | 154.76 | 1.00x | 124.46 / 123.34 | OK/OK |
| 8 | 9 | 14534.06 | 229.67 | 63.28x | 156.44 | 157.01 | 1.00x | 124.60 / 124.50 | !/! |
| 8 | 16 | 25758.36 | 206.00 | 125.04x | 158.25 | 156.56 | 1.01x | 128.49 / 124.70 | !/! |
| 8 | 32 | 1229.66 | 257.58 | 4.77x | 253.23 | 142.82 | 1.77x | 138.12 / 137.51 | OK/! |
| 8 | 64 | 2435.23 | 478.10 | 5.09x | 448.44 | 189.99 | 2.36x | 212.08 / 210.72 | OK/OK |
| 8 | 128 | 4826.07 | 917.02 | 5.26x | 870.48 | 330.16 | 2.64x | 383.75 / 382.32 | OK/OK |
| 8 | 512 | 17153.68 | 2982.80 | 5.75x | 3356.11 | 1240.39 | 2.71x | 1395.58 / 1399.89 | OK/OK |

#### Dispatch evidence

Separate Nsight Systems 2026.3.2 captures used K=N=4096, W2/W4/W8,
M=1/16/128, and both dynamic layouts. Each NVTX range contains ten graph
replays after ten warmups; these profiler runs were not used for the timing
tables above. SQLite exports retain demangled kernel names, register
counts, launch geometry, and durations.

Selected **matmul-only** median durations, excluding the activation
quantizer and host gaps:

| GPU | Case | Before kernel / grid CTAs | After kernel / grid CTAs | Before / after us |
|-----|------|---------------------------|--------------------------|-------------------|
| RTX 5060 Ti | W2 M1 standard | `IntegerMatMul` / 512 | aligned grouped, `16 x 32` / 128 | 239.357 / 26.992 |
| RTX 5060 Ti | W4 M16 standard | `IntegerMatMul` / 8192 | fine aligned grouped, `16 x 32` / 128 | 6870.056 / 43.520 |
| RTX 5060 Ti | W4 M128 standard | `NativeIntegerMatMul` / 256 | fine aligned grouped, `64 x 64` / 128 | 1137.220 / 210.558 |
| RTX 5060 Ti | W4 M128 prepacked | one row fragment / 512 | four row fragments / 128 | 154.319 / 89.679 |
| RTX 5060 Ti | W8 M128 prepacked | one row fragment / 512 | four row fragments / 128 | 170.174 / 94.703 |
| RTX 3060 | W2 M16 standard | `IntegerMatMul` / 8192 | aligned grouped, `16 x 32` / 128 | 7300.236 / 67.663 |
| RTX 3060 | W4 M128 standard | `NativeIntegerMatMul` / 256 | fine aligned grouped, `32 x 64` / 256 | 2431.588 / 511.997 |
| RTX 3060 | W4 M128 prepacked | one row fragment / 512 | two row fragments / 256 | 316.014 / 201.343 |
| RTX 3060 | W8 M128 prepacked | one row fragment / 512 | four row fragments / 128 | 418.094 / 170.783 |

The aligned grouped launches resolve to
`NativeAlignedGroupedMatMulCapped`; its final template flag selects the
fine-interval variant for W4/W8 block-32. Prepacked launches resolve to
`PrepackedMatMul` with the expected row-fragment template argument.
All 18 cases on each GPU retained the same `QuantizeRowsVector` kernel
name, grid, block size, and register count before/after. These captures
verify routing and launch structure, not clock-independent performance.

#### Correctness, build, and reproduction

The full existing Python operator test module passed on both GPUs:
**20 tests each, zero skips** (35.156 s on RTX 5060 Ti and 34.002 s on
RTX 3060). The added regression compares exact FP16 reference bits for
standard/prepacked W2/W4/W8, per-token/per-tensor quantization, bias,
zero activation rows, and changed activations during CUDA graph
capture/replay. Its M values 1/8/9/16/32/64/65/128/129 exercise row-fragment
and grouped-tile boundaries. Existing tests retain coverage of the other
operator modes, data types, scale policies, and unsupported geometry.
All benchmark dynamic output hashes also match the frozen baseline.

For the random benchmark inputs, relative RMS output difference against
the weight-only control is 0.008056..0.010239 on RTX 5060 Ti and
0.008058..0.010239 on RTX 3060. This is not a model-level accuracy
evaluation: weight-only `MatMulNBits` has different numerical semantics
and is neither an exact dynamic reference nor universally slower.

The initial bundled build passed using the following command from a
VS2022 x64 compiler environment with `D:\py312` activated:

```powershell
python tools\ci_build\build.py --cmake_generator "Visual Studio 17 2022" `
  --config Release --build_dir build\sm120 --build_wheel --update --build `
  --parallel 8 --nvcc_threads 1 --build_shared_lib --use_cuda --cuda_version 13.0 `
  --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" `
  --cudnn_home "D:\cudnn\9.13.0.50_cuda13" --skip_tests --skip_submodule_sync `
  --use_binskim_compliant_compile_flags `
  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=86-real;120-real" `
    onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_BUILD_UNIT_TESTS=OFF
```

The incremental optimized build also passed:

```powershell
python tools\ci_build\build.py --config Release --build_dir build\sm120 `
  --build --target onnxruntime_providers_cuda --parallel 8 --nvcc_threads 1 --skip_tests
```

The build retains an existing `LNK4098` runtime-library conflict warning.
C++ unit-test binaries were not built; the full Python operator suite was
run against the rebuilt CUDA provider. The plugin CUDA EP build was not
validated by this measurement.

Machine-local frozen runtimes, scripts, raw results, test/build logs,
Nsight captures, and SQLite exports are retained under:

```text
C:\Users\tlwu\.copilot\session-state\9a51c884-e203-4804-b4d8-601a5df1ed64\files
```

The task-local virtual environment adds the existing
`D:\py312\Lib\site-packages` dependency directory through a `.pth` file;
only the missing `onnx_ir` package was installed into that isolated
environment. No shared Python package was replaced.

The benchmark defaults encode the full sweep above. After rebuilding and
freezing matching before/after runtimes, use fresh output names rather
than overwrite the evidence:

```powershell
$art = 'C:\Users\tlwu\.copilot\session-state\9a51c884-e203-4804-b4d8-601a5df1ed64\files'
$py = "$art\bench_venv\Scripts\python.exe"
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;D:\cudnn\9.13.0.50_cuda13\bin;' + $env:PATH
Set-Location D:\git\dev
$env:CUDA_VISIBLE_DEVICES = '0' # '1' for RTX 3060
$env:PYTHONPATH = "$art\baseline_rt"
& $py "$art\bench_checked.py" --out "$art\rerun_baseline_5060_1.json"
$env:PYTHONPATH = "$art\optimized_rt"
& $py "$art\bench_checked.py" --plan "$art\rerun_baseline_5060_1.json" `
  --reference "$art\rerun_baseline_5060_1.json" --out "$art\rerun_after_5060_1.json"
```

Recorded results: all original three-pass measurements contain 48 complete
cases per GPU/phase/pass with exact dynamic bit/hash parity. Repeat both
phases with the same `--plan` and `--reference` for passes 2 and 3.
The original aggregation command passed:

```powershell
& $py "$art\summarize_results.py"
```

It reads `baseline_<gpu>_<pass>.json` / `after_<gpu>_<pass>.json` and
writes `comparison.json` / `comparison.txt`. For reruns with different
file prefixes, update the artifact-only aggregator's input names.

The full unittest launcher imports Torch and the selected frozen ORT first,
then loads the source test module by file path. Running from outside the
repository avoids accidentally importing the source-only ORT package:

```powershell
Set-Location D:\git\dev
$env:PYTHONPATH = "$art\optimized_rt"
$env:CUDA_VISIBLE_DEVICES = '0'
& $py "$art\run_operator_tests.py" # Passed: 20 tests, zero skips
$env:CUDA_VISIBLE_DEVICES = '1'
& $py "$art\run_operator_tests.py" # Passed: 20 tests, zero skips
```

Python `ruff check` and `ruff format --check` passed for the changed test
module; clang-format 18.1.8 checks passed for the changed C++ line ranges.
The exact Python checks are:

```powershell
D:\py312\Scripts\ruff.exe check onnxruntime\test\python\contrib_ops\test_dynamic_quantize_matmul_nbits.py
D:\py312\Scripts\ruff.exe format --check onnxruntime\test\python\contrib_ops\test_dynamic_quantize_matmul_nbits.py
```

Run these lint commands from the repository root. For dispatch reproduction,
set the selected runtime/GPU as above and run the following with a fresh
output prefix; original captures and exports passed on both GPUs:

```powershell
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe'
& $nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none `
  --cuda-graph-trace=node -o "$art\rerun_profile_after_3060" `
  $py "$art\profile_checked.py"
& $nsys export --type=sqlite --output="$art\rerun_profile_after_3060.sqlite" `
  "$art\rerun_profile_after_3060.nsys-rep"
```

`profile_report.py` aggregates the original `profile_<phase>_<gpu>.sqlite`
files into `profile_comparison.json` and checks ten kernel samples per
named range.

#### Interpretation and limitations

- The large standard-layout gains primarily eliminate poorly matched DP4A
  or synchronous native dispatch, rather than speed up activation
  quantization. Prepacked gains at larger M primarily come from reuse of
  decoded weights across row fragments. W8 at M <= 16 is essentially
  unchanged; modest small-M W2/W4 improvements are not equally stable on
  both GPUs.
- No dynamic median regression exceeds 1%. There are approximately 0.9%
  and 0.6% slower standard W2 cases on RTX 5060 Ti / RTX 3060 respectively,
  even with `OK` quality flags; these are retained in the tables. No claim
  is made that every case improves. The small prepacked W8 regressions on
  RTX 3060 fail the strict timing-quality criteria.
- The unchanged weight-only control has geomean B/A approximately 1.006
  on both devices. Individual control ratios range from 0.976 to 1.054 on
  RTX 5060 Ti and 0.993 to 1.030 on RTX 3060; many control cases also fail
  the strict quality thresholds. The control is a scheduling/clock-drift
  check, not evidence of an optimization to `MatMulNBits`.
- These are live Windows WDDM measurements with unlocked clocks. RTX 3060
  also drives the desktop, and its before/after spread is appreciably less
  stable. Our timing and profiling GPU workloads ran sequentially, but
  desktop activity and other shared-machine effects were not eliminated.
  `!` results should not be treated as precise speedup claims.
- The sweep covers the listed block sizes, two K/N shapes, M <= 512,
  FP16, and per-token activation quantization. Correctness tests cover
  additional contracts; performance on other shapes, modes, larger GPUs,
  and the plugin CUDA EP remains unmeasured here.

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
