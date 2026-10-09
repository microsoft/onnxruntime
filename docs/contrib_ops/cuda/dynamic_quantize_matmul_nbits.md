# DynamicQuantizeMatMulNBits - CUDA Operator Design

**Status:** Proposed; this document does not describe an implemented operator.

**Motivation:** [ONNX Runtime issue #33200](https://github.com/microsoft/onnxruntime/issues/33200)
requests CUDA GEMM with dynamically quantized per-token INT8 activations,
per-channel INT8 weights, and a floating-point output with scaling fused into
the GEMM epilogue.

The proposed operator follows the architecture of
[MatMulBlockQuantizedFp4Weight](./matmul_block_scaled_fp4.md) and
[MatMulBlockQuantizedFp8Weight](./matmul_block_scaled_fp8.md), reusing their
tensor layout, block-scale representation, optional bias, and separation of
operator semantics from CUDA dispatch. Activations are always dynamically
quantized to symmetric INT8 inside the operator. Weight-only execution
remains the responsibility of MatMulNBits and the existing FP4/FP8 operators;
every backend of the initial INT8-weight implementation must preserve W8A8
integer-dot-product semantics. The name is weight-bit-width independent so
that future 4-bit and 2-bit integer weights can use the same operator family.
`NBits` refers to weight precision; activation precision remains INT8.

## 1. Goals and Non-Goals

Goals:

- Accelerate prefill and encoder GEMMs with symmetric per-token activation
  quantization and per-channel weight quantization.
- Support per-tensor, per-token, and per-token K-block dynamic activation
  quantization through one explicit model-level granularity attribute.
- Generalize the weight representation to scales that vary along K.
- Support FP16 and BF16 activations, bias, and output.
- Fuse scaling and bias into the native GEMM output stage, avoiding a global
  INT32 output intermediate on the primary fast path.
- Provide correct small-M and general CUDA paths without changing the
  model's selected activation-quantization mode.
- Keep weights and the serialized operator contract architecture independent.
- Reserve a weight-bit-width attribute so lower-bit integer weights can be
  added without renaming the operator or reinterpreting existing INT8 models.

Non-goals for the initial implementation:

- Implementing 4-bit or 2-bit weights; the initial supported weight width is 8 bits.
- UINT8 or asymmetric quantization, nonzero zero points, or arbitrary group indices.
- Weight-only W8A16 execution, static activation scales, or externally
  quantized activation inputs.
- FP32 activation/output support or quantized output.
- Batched weight matrices or general transpose attributes.
- Automatically enabling W8A8 for ordinary floating-point MatMul nodes.
- Changing existing FP4, FP8, or MatMulNBits operator behavior.

SmoothQuant or other outlier-management transformations belong in model
preparation, not in an implicit runtime heuristic.

## 2. Operator Schema

Proposed name: `com.microsoft::DynamicQuantizeMatMulNBits`, version 1.
The initial schema and tensor formats below support `bits = 8` only.

| Input | Index | Type | Shape and meaning |
|-------|-------|------|-------------------|
| `A` | 0 | FP16 or BF16 (`T`) | Row-major activation, `[..., K]`, rank at least 1. |
| `B` | 1 | INT8 (`T1`) | Row-major symmetric quantized weight, `[N, K]`. |
| `b_scale` | 2 | FP32 (`T2`) | Weight scales, `[N, G]`, where `G = ceil(K / block_size)`. |
| `bias` | 3 | Optional `T` | Per-output-channel bias, `[N]`. |

| Attribute | Type | Default | Meaning |
|-----------|------|---------|---------|
| `bits` | INT | `8` | Integer weight bit width. Initially only `8` is supported; `4` and `2` are reserved for future implementation. |
| `block_size` | INT | `128` | Positive number of consecutive K elements sharing a weight scale. |
| `activation_quantization` | STRING | `"dynamic_per_token"` | `"dynamic_per_tensor"`, `"dynamic_per_token"`, or `"dynamic_per_block"`. All modes dynamically quantize A to symmetric INT8. |
| `activation_block_size` | INT | No default | Required and positive for `"dynamic_per_block"`; must be absent for the other modes. |

Output `Y` has type `T` and shape `[..., N]`. Bias must have the same type as A.
Scales are dequantization multipliers, not their reciprocals.

The weight zero point is implicitly zero. All INT8 weight values, including
`-128`, are supported. Generated activation codes use the symmetric range
`[-127, 127]`.

The schema deliberately omits an `a_scale` input. Dynamically generated activation
scales are internal to the operator; optional-input presence must not
implicitly select a different quantization mode.
There is no `"none"` mode. `DynamicQuantize` makes mandatory activation
quantization explicit, while `MatMulNBits` identifies the extensible integer
weight representation. It is distinct from the weight-only MatMulNBits operator.

### 2.1 Per-channel weights are a single K block

For `K > 0`, set `block_size = K` to obtain one scale per output channel:

```text
A:       [M, K]      FP16/BF16
B:       [N, K]      INT8
b_scale: [N, 1]      FP32
bias:    [N]        FP16/BF16, optional
Y:       [M, N]      FP16/BF16
```

Any positive block size with `G == 1` represents per-channel weights.
It is eligible for the final-epilogue-only fast path when the selected
activation mode also has no scale changes along K.
Do not require a power-of-two block size: K itself need not be a power
of two. Using this single-block representation avoids a new sentinel value
or a separate weight-granularity attribute.

An exporter can reshape existing `[N]` per-channel scales to `[N, 1]` without
changing their values. Version 1 accepts only the canonical two-dimensional
scale shape, not implicit broadcasting of scalar or one-dimensional scales.

### 2.2 Block-scaled weights

For positive block size S:

```text
G = K / S + (K % S != 0)
g(k) = k / S
B_scaled[n,k] = float(B[n,k]) * b_scale[n,g(k)]
```

The last block may be shorter than S. Physical padding, if needed by a CUDA
backend, is internal and must not affect the serialized shapes or reductions.

### 2.3 Shape inference and validation

Derive N and K from B; do not duplicate them in attributes. Propagate A's
element type, preserve all its leading dimensions, and replace its last
dimension with N. A rank-1 input `[K]` produces `[N]`.

Schema inference and runtime validation must check all available information:

- A has rank at least 1 and B has rank 2.
- A's last dimension equals B's second dimension.
- `bits == 8` in the initial implementation. Reserved or otherwise unsupported
  widths produce an explicit error, not reinterpretation as INT8.
- `block_size > 0` and the activation mode is recognized.
- `activation_block_size > 0` is present exactly when the mode is
  `"dynamic_per_block"`.
- `b_scale` has rank 2 and shape `[N, G]`.
- Bias, when present, has shape `[N]` and type T.
- Dimensions and scratch sizes fit the backend's index and allocation types.

Unknown or symbolic dimensions must remain valid during shape inference.
Use checked dimension and byte-size arithmetic at runtime, including padded
dimensions and rounding to backend alignments.

### 2.4 Future lower-bit weight extension

Keep activation modes, activation-scale rules, weight block scales, bias,
and output semantics independent of `bits`. Existing models that omit
`bits` continue to use 8-bit weights.

Before enabling `bits = 4` or `bits = 2`, the extension must define:

- Packed storage type and shape, signed code encoding, bit order, and tail padding.
- Recovery of logical K from A and validation against the packed weight shape;
  the INT8-only rule `K = B.shape[1]` cannot be reused for packed weights.
- Unpacking and integer-dot backends, with the same common-scale-interval rules.
- Accumulator bounds for the supported signed weight range.
- Export, prepacking, and tests keyed by the actual weight bit width.

The existing `[N, K]` INT8 format must not change when packed formats are
introduced. A versioned schema extension must preserve the `bits = 8` contract and
explicitly describe any new input types and layouts. Reserving attribute
values does not claim that their formats or kernels are already supported.
Do not retroactively reinterpret version-1 inputs when adding packed types.

## 3. Numerical Contract

Flatten A's leading dimensions into M independent rows. This matches the
existing [MatMulComputeHelper](../../../onnxruntime/core/providers/cpu/math/matmul_helper.h)
handling of `[..., K]` multiplied by a transposed two-dimensional weight.

Weight scales must be positive and finite; report other values as
`INVALID_ARGUMENT`, never replace them with a default. Section 5 specifies
when scale values are validated and the resulting graph-capture restriction.

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
acc[m,n] = sum_k int32(qA[m,k]) * int32(B[n,k])
scaled[m,n] = ScaleProduct(acc[m,n], sA_for_row[m], b_scale[n,0])
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
and `b_scale = float32(1e-38)` have a finite result near 13.56. Multiplying
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
acc_p[m,n] = sum_{k in interval p} int32(qA[m,k]) * int32(B[n,k])
term_p    = ScaleProduct(acc_p[m,n], sA[m,p], sB[n,p])
```

Here `sA[m,p]` and `sB[n,p]` select the original scales covering the interval;
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

Grouping is defined by the schema, not the CUDA tile shape. Do not merge
adjacent intervals merely because their scale values happen to be equal:
that can change the specified FP32 rounding of partial sums.

### 3.5 Accumulator limits and empty dimensions

With activation codes bounded by 127 and weight magnitudes bounded by 128,
an input-independent INT32 safety bound for one common scale interval is:

```text
S_A = activation_block_size for dynamic_per_block, otherwise K
L_bound = min(K, block_size, S_A)
L_bound * 127 * 128 <= INT32_MAX
```

Validate the bound using checked wider arithmetic. Version 1 reports
`INVALID_ARGUMENT` when this conservative bound is exceeded; do not wrap or
use saturating integer accumulation. It applies to each partial accumulator,
not the entire K when scales subdivide the contraction.
Supporting longer intervals with a wider exact accumulator
would require an explicitly specified extension.

An empty output returns without launching compute. For `K == 0`, the empty
sum is zero and Y is zero plus optional bias; no row quantization or GEMM is
needed. With the canonical scale representation, `b_scale` has shape `[N, 0]`
in this case, and `block_size` remains positive.

## 4. CUDA Dispatch

Dispatch selects an implementation, never the activation-quantization mode.
Using the new operator is the model's W8A8 opt-in; the serialized attribute
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
            +-- one interval, native ready --> INT8 tensor-core GEMM + epilogue
            +-- multiple intervals, ready ---> interval-partial INT8 GEMM
            +-- otherwise ------------------> general integer CUDA fallback
```

Fast-path eligibility includes the compiled kernel's availability, device
capability, dtype, alignment, dimensions, and scale layout. A device-major
comparison alone is not sufficient evidence that a native kernel is available.

Unsupported fast-path geometry routes to the generic implementation.
An actual launch, allocation, or CUDA-library error is returned to ORT;
do not hide an execution failure by retrying another backend.

### 4.1 Activation quantization

Share the quantization rule and conversion helpers across native, small-M,
and fallback paths. Produce:

- `qA`: `[M, K_padded]` INT8, with zero-filled internal K padding.
- `sA`: scalar, `[M]`, or `[M, G_A]` FP32, according to the selected mode.

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

### 4.3 Small-M and block-scaled kernels

For decode and short speculative batches, use integer GEMV or a smaller
GEMM when measured latency is lower. Reuse qA and sA. A DP4A/scalar integer
path is a valid baseline; choosing small M must not bypass activation
quantization or change its granularity.

For multiple scale intervals, reset integer accumulation whenever either
operand's scale changes and retain a running FP32 output accumulator.
Support arbitrary positive activation and weight block sizes in the generic
path, even if native tiles require sizes aligned to their MMA K extent or
matching block boundaries. Per-channel weights alone do not make
multi-block activations eligible for a final-epilogue-only kernel.

Do not reorder FP32 block reductions or use nondeterministic atomic
split-K accumulation in a backend advertised as bit-exact.

### 4.4 Correctness fallback

W8A8 fallback choices are:

1. A direct CUDA integer-dot kernel with the specified scaling order.
2. INT8 GEMM into a bounded INT32 tile, followed by a scale/bias kernel.
3. For multiple intervals, integer partial GEMMs with FP32 interval accumulation
   and a final bias/conversion step.

The existing
[GemmInt8 helper](../../../onnxruntime/core/providers/cuda/integer_gemm.cc)
is potential reusable infrastructure. It currently consumes non-transposed
logical B in `[K, N]` layout, so it cannot directly consume this operator's
`[N, K]` storage. Reuse requires a transpose-aware extension or explicit
internal repacking and validation of CUDA-library restrictions.

**Dequantize to FP16/BF16 followed by floating GEMM is not an exact W8A8
fallback.** It rounds intermediate values and changes accumulation semantics.
It is not a backend of this dynamic-only operator.

## 5. Memory, Prepacking, and Streams

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

Use `PrePack` only for constant weights/scales that need backend-specific
padding, transposition, or reordering. Such formats remain internal:

- Do not serialize an Ada-specific or Blackwell-specific layout in B.
- Preserve a usable representation for fallback backends.
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

- Validate non-overridable constant `b_scale` initializers once during session
  initialization. If device-resident, a validation kernel and a checked
  device-to-host flag transfer may synchronize the initialization stream.
  Preserve the validated initializer or its packed representation.
- For runtime or overridable `b_scale`, validate every invocation with
  nonempty computation. A checked validation flag transfer and compute-stream
  synchronization are allowed outside capture, and must be included in
  end-to-end benchmarks for runtime scales. Do not use device-wide synchronization.
- CUDA graph capture requires an already validated, non-overridable constant
  `b_scale`. Reject runtime/overridable scales during capture before issuing
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
  weight-only contract. Keep W8A16 there rather than duplicating it in the
  new operator. It must not start dynamically quantizing activations based
  on runtime shape or GPU throughput.
- Existing FP4/FP8 nodes retain their current semantics and input positions.

Expose per-node opt-in through quantization/export tooling. For example:

```python
node = onnx.helper.make_node(
    "DynamicQuantizeMatMulNBits",
    ["A", "B", "b_scale", "bias"],
    ["Y"],
    domain="com.microsoft",
    bits=8,
    block_size=6144,
    activation_quantization="dynamic_per_token",
)
```

For this example, B is `[N, 6144]` and `b_scale` is `[N, 1]`. Omit the
fourth input when bias is absent. The model must import version 1 of the
`com.microsoft` domain once the proposed operator is implemented.

To use a single dynamic scale for all A values, select
`activation_quantization="dynamic_per_tensor"` instead. For per-token K-block
activation quantization, select:

```text
activation_quantization = "dynamic_per_block"
activation_block_size = 128
```

The activation block size is independent of `block_size=6144`: this example
has per-channel weights and 48 activation scales per token. It needs
scaled interval partial sums, not the per-channel final-epilogue-only kernel.

Do not fuse a floating MatMul or the existing UINT8 dynamic-quantization
pattern into this node unless the graph already explicitly selects the
same symmetric granularity, ranges, and rounding semantics. Node-level selection is necessary
because activation outliers and smoothing requirements are model dependent.

If multiple GEMMs share A, independently quantizing A is valid but may repeat
work. A reusable dynamic-quantization operator and an externally quantized
GEMM can be considered separately; they are not prerequisites for this design.
Do not share a quantized activation or scale across nodes with different
granularities or activation block sizes.

## 7. Implementation Surfaces

The implementation should follow the neighboring FP4/FP8 operator structure:

| Surface | Required work |
|---------|---------------|
| [Contrib schemas](../../../onnxruntime/core/graph/contrib_ops/contrib_defs.cc) | Add schema documentation, type constraints, attributes, and shape inference. |
| [CUDA math operators](../../../onnxruntime/contrib_ops/cuda/math/) | Add `dynamic_quantize_matmul_nbits.cc` and `.h` for the operator and dispatch; use `matmul_block_scaled_int8.cu` for the initial INT8 backend. |
| [CUDA contrib registration](../../../onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc) | Add declaration and kernel-create registration. |
| [Standard CUDA build](../../../cmake/onnxruntime_providers_cuda.cmake) and [plugin CUDA build](../../../cmake/onnxruntime_providers_cuda_plugin.cmake) | Include the same operator and correctly gated native kernels in both builds. |
| [Shared CUDA source filtering](../../../cmake/onnxruntime_cuda_source_filters.cmake) | Isolate architecture-specific sources where needed; retain generic kernels when native sources are excluded. |
| [Quantization tooling](../../../onnxruntime/python/tools/quantization/) | Add explicit per-node export selection and canonical weight/scale conversion. |
| [Contrib operator tests](../../../onnxruntime/test/contrib_ops/) | Add focused OpTester cases and direct quantization/dispatch tests. |
| [Contrib benchmark harness](../../../onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py) | Extend or reuse the existing accuracy/latency harness for INT8 cases. |
| [Contrib operator documentation](../../../docs/ContribOperators.md) | Regenerate the public operator entry after schema implementation. |

Do not make the new operator depend on FP4/FP8 type support or optional MoE
build flags. Windows/MSVC, Linux, and plugin builds need the generic operator
even when a particular native backend is not compiled.

## 8. Validation Plan

### 8.1 Numerical and shape coverage

- FP16/BF16; rank-1, rank-2, and higher-rank A; optional bias.
- All three dynamic modes combined with per-channel and block-scaled weights.
- Unequal activation/weight block sizes, misaligned boundaries, and
  deliberately unequal scales on consecutive intervals.
- INT8 weights including `-128` and `127`; negative activations and outputs.
- All-zero rows, very small finite rows, large finite values, and RNE ties.
- Extreme BF16/scaling combinations that overflow the first FP32 product
  despite a finite final result, plus the guarded wider branch for subnormals.
- Quantization values immediately on both sides of a rounding boundary.
- M/N/K tails, partial final blocks, and non-power-of-two block sizes.
- Empty M/N, `K == 0`, and accumulator-bound acceptance/rejection.
- Invalid ranks, dimensions, scale shapes, attributes, and bias types/shapes.
- Explicit errors for zero, negative, NaN, or infinite weight scales,
  including an invalid runtime scale supplied after a valid invocation.
- NaN/infinite activations in each granularity, with the specified affected
  rows and no dependency on NaN payload bits; nonfinite bias propagation.
- Default and explicit `bits = 8` equivalence; explicit rejection of
  reserved 4-bit/2-bit weights until their formats and backends are implemented.
- Rejection of the removed `"none"` mode and missing/extra activation block
  size attributes.
- Batch independence for per-token/per-block, including an extreme outlier
  added in another row; intentional shared-scale changes for per-tensor.
- Per-tensor parity when M is scratch-tiled, proving its scale is still global.
- Per-block locality: changing another activation block must not change the
  unchanged block's scale or quantized values.

Build the W8A8 reference from the actual FP16/BF16 input values, not their
pre-conversion FP32 sources. Compute integer dot products in a wider host
type, check the supported bound, and apply the specified FP32 ordering.
Test qA codes exactly; output closeness alone can miss quantization mistakes.

For forced W8A8 backends, require identical activation codes, integer partial
sums, and output bits against the reference for finite-domain test cases.
Exercise both branches of `ScaleProduct` and FP16/BF16 output rounding.
Use the common-interval reference, not an assumption that activation and
weight block indices are interchangeable.

### 8.2 Dispatch and execution coverage

- Force native, small-M, blocked, and generic paths through internal test
  controls without adding public model attributes.
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
weights across all three activation granularities. Validate on Ada and the
reported SM121 workload, with representative
Ampere/Hopper/Blackwell coverage where available. Do not assume identical
kernel eligibility or crossover points across those architectures.

Report reduction/quantization, GEMM, fallback scaling, and total operator latency.
Identify constant versus runtime scales and include runtime-scale validation
overhead. Measure the guarded wider epilogue separately on extreme-value cases;
it must not force the ordinary-value workload onto an FP64 GEMM.
Measure cuBLASLt INT8 plus its required scaling/output conversion, FP16 GEMM,
and MatMulNBits with clearly identified quantization settings. A comparison
against bare INT8-to-INT32 GEMM alone is not an end-to-end comparison.

Include warmup, repeated CUDA-event measurements, ORT session latency,
workspace usage, GPU/toolkit versions, and actual dispatch. Issue-reported
speedups are motivation, not verified performance guarantees for this proposal.
Model-level accuracy must be checked with the chosen per-node configuration,
including an outlier-sensitive case with and without offline smoothing.

## 9. Delivery Plan and Acceptance

1. Implement the proposed schema, numerical-input policies, and shape validation,
   the three dynamic quantization granularities, and generic integer paths
   for both single and multiple common scale intervals.
2. Integrate the per-channel tensor-core backend and fused scale/bias epilogue
   for activations with no scale changes along K. Keep general interval
   scaling correct even before it is performance tuned.
3. Add measured small-M dispatch and optimize the multi-interval backend
   without changing quantization domains or interval-reduction semantics.
4. Wire explicit model export, standard/plugin build coverage, documentation,
   and end-to-end accuracy/performance validation.

The first advertised version must have a working fallback for every
schema-supported weight/activation block size and dynamic mode. If a
preliminary PR only implements the per-token, single-weight-block case,
label and reject unsupported configurations explicitly rather than claiming
support for other granularities.

Acceptance requires quantization-reference parity, semantics-preserving
dispatch, bounded scratch, capture/concurrency correctness under the documented
constant-scale capture restriction, both CUDA build
surfaces, and measured end-to-end benefit on the target prefill workloads.
The core design invariant is:

**The operator always dynamically quantizes activations. The model selects
the granularity; CUDA dispatch selects only an implementation of that contract.**
