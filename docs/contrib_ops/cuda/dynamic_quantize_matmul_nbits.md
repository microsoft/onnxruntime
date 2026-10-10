# DynamicQuantizeMatMulNBits - CUDA Operator Design

**Status:** Implementation in progress: schema, generic DP4A path, native
INT8 tensor-core dispatch with direct INT8/INT4/INT2 decoding, all three
activation granularities, and explicit default-quantizer export opt-in.
Aligned native paths use asynchronous packed-weight/activation staging:
two stages for single-interval W8 and three for grouped weights.
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
- Reusing architecture-specific `weight_prepacked=1/2` MatMulNBits layouts;
  version 1 consumes its standard, non-prepacked representation only.
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
| `B` | 1 | UINT8 (`T1`) | Standard MatMulNBits blobs, `[N, G, blob_size]`, where `G = ceil(K / block_size)` and `blob_size = block_size * bits / 8`. |
| `scales` | 2 | FP16, BF16, or FP32 (`T2`) | MatMulNBits weight scales, `[N, G]` or flattened `[N * G]`, in output-channel/block order. |
| `bias` | 3 | Optional `T` | Per-output-channel bias, `[N]`. |

| Attribute | Type | Default | Meaning |
|-----------|------|---------|---------|
| `bits` | INT | `8` | Integer weight bit width; `8`, `4`, and `2` are supported by the proposed contract. Other widths are rejected. |
| `block_size` | INT | `128` | Number of logical K elements sharing a weight scale; power of two and at least 16, as in MatMulNBits. For `bits = 2`, at most 256. |
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
This does not include asymmetric zero points, `g_idx`, or CUDA
`weight_prepacked=1/2` layouts. QServe, GGUF, and Marlin formats are not
implicitly accepted. Internal prepacking must preserve the decoded
logical codes and scale values.
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
- Other native W8A8/W4A8/W2A8: `32 x 64 x 64` CTA tile with direct shared-memory
  weight decoding. Requires M at least 32, N at least 8, K at least 32,
  and every interior activation/weight block boundary 32-aligned: each
  block size must be at least K or divisible by 32.

M/N tails and final partial K tiles are predicated. Unsupported geometry
uses the one-warp-per-output DP4A/scalar path; all paths accumulate common
intervals in the same ordered FP32 sequence.

The single-interval M threshold was checked against DP4A on the measured
H200 shapes, but is not universally tuned. Aligned pipeline tile choices
were measured on H200; cross-architecture crossovers remain unverified.
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
Reject nondefault zero points, `g_idx`, and prepacked layouts rather than
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
