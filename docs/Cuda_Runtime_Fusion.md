# CUDA runtime fusion

## Unit-Swish

The internal session entry `ep.cuda.runtime_fusion.unit_swish` enables the
`unit_swish_scaled_silu_v1` Rama rule. It is off by default and accepts only
the exact strings `"0"` and `"1"`.

The rule replaces standard ONNX Swish-24 with absent or FLOAT alpha=1 by the
existing `com.microsoft.ScaledSiLU-1`, with alpha=1 and no scale input.
Input and output must be dense, matching FLOAT or BF16 tensors with compatible
shape and initializer metadata. FLOAT16, DOUBLE, mixed types, non-unit alpha,
and nonfinite alpha are not admitted. No casts, schema changes, or target
kernel changes are introduced.

### Discovery and acceptance

Discovery requires a capability consumer that executes optimization callbacks,
optimization level 2 or higher, ONNX-24/Microsoft-1 imports, an unfiltered main
graph, normal CUDA layout, and no resource accountant. Minimal CUDA,
contrib-disabled, and plugin CUDA builds do not offer this optimization.
ORT-format loading does not execute these callbacks and opts out.

Discovery reads metadata only, including for initializer inputs: it does not
open external files or inspect tensor payloads. A scratch graph probes the
exact typed target kernel. A missing target leaves ordinary execution eligible.
Symbolic, anonymous, scalar, and empty shapes are eligible. Known element counts
must fit the target's 256-thread grid with a signed-int block count; unknown
extents remain subject to runtime kernel validation.

Acceptance revalidates the live source, assignment, attributes, NodeArg
identities, and owned type/shape snapshot. A region-restricted Rama `Run` over
the original single node must perform exactly one rewrite to one target.
The original input and output NodeArgs, external fanout, and graph outputs
are preserved. Stale offers fail rather than silently selecting another source.
Changing an input's producer alone does not invalidate its NodeArg snapshot.

### Numerical scope and tests

At unit alpha, the source CUDA Sigmoid computes in FLOAT before converting to
the element type, and the element product converts back to that type. The
target also evaluates its stable sigmoid in FLOAT, with element conversions.
Non-unit alpha is not covered.

Focused source/fused/direct tests require identical non-NaN bits (including
signed zero) and matching NaN classifications, not matching NaN payloads.
They cover scalar, empty, odd, dynamic, and model-shaped inputs. FLOAT cases
use pinned special values and seeded random raw bits. BF16 coverage includes
all 65536 bit patterns in one rank-1 case; other shapes use pinned special
values and seeded random bits. Graph and profile assertions verify CUDA
placement and the replacement; values alone are insufficient evidence.
These tests are not a guarantee for every input, shape, dtype, device, model,
or performance regime.

Run freshly built tests from the build directory:

```sh
./onnxruntime_test_all --gtest_filter='UnitSwishHostTest.*'
./onnxruntime_provider_test --gtest_filter='CudaRuntimeFusionTest.UnitSwish*'
```

Confirm nonzero counts and inspect skips. CUDA numerical tests require a
supported device; BF16 tests check device support before constructing test
fixtures. Qualification logs, model counts, and hardware timings belong in
validation artifacts rather than this contract.

## Stateful causal convolution

The internal session entry `ep.cuda.runtime_fusion.causal_conv_with_state`
enables the `causal_conv_with_state_mobius_bf16_v1` Rama rule. Despite its
historical identifier, the rule supports matching FLOAT, FLOAT16, or BF16
regions. It is off by default and accepts only the exact strings `"0"` and
`"1"`. The discovery-consumer, optimization-level, import, layout, and build
restrictions described above also apply.

### Mathematical contract

The rule replaces a channels-first causal depthwise convolution region:

- Concatenate history `[B,C,(K-1)*D]` with input `[B,C,S]` along axis 2.
- Apply an unpadded, stride-1 Conv with constant weight `[C,1,K]`,
  group `C`, positive dilation `D`, and a constant per-channel bias.
- Apply unit-alpha Swish and return its output and the final history tail.

Bias may be a Conv input, a broadcast `[1,C,1]` Add, or a `[C]` initializer
reshaped before Add. The history tail must use the recognized Shape/Gather/
Sub/Reshape/Slice construction with the correct axis and window. The
normalized empty tail is supported for `K=1`. Shared internal results that
would be removed cannot escape the matched region.

The existing target is `com.microsoft.CausalConvWithState-1`, with
`ndim=1`, `channels_last=0`, `state_window=0`, the proven dilation, and
`activation="silu"`. Its mathematical Conv/bias/SiLU and history semantics
must match the source. It does not reproduce every source floating-point
materialization boundary: accumulation order, intermediate rounding, and
overflow/NaN/Inf behavior may differ. In particular, a source Conv may
overflow its element type before Swish, whereas the target can retain a
finite FP32 accumulator through SiLU. Opting in accepts this numerical
contract, not changes to tensor layout, bias, axes, or state updates.

### Discovery and acceptance

Discovery checks eligible, unassigned Swish roots before invoking Rama.
Channels, kernel size, dilation, shapes, control constants, initializer
storage, and native index/byte bounds are checked from the matched region.
Small control constants must be embedded. Learned weight and bias payloads
are not read and external files are not opened during discovery. Unknown
runtime dimensions are not replaced with guessed workload sizes.

A scratch graph probes the exact typed native target. Missing kernels or
unsupported contracts leave ordinary execution eligible. Discovery does not
mutate the source graph. Acceptance revalidates the exact offered region and
contract, rewrites it once, preserves both original output NodeArgs, and
resolves the graph. A folded rank-3 bias receives an owned, flattened bitwise
copy without reshaping shared source storage. Stale acceptance or target
construction errors fail initialization rather than silently changing rules.

The causal rule is offered before unit-Swish when both flags are enabled.
Accepted causal regions own their contained Swish; other eligible Swish
nodes remain available to the independent unary rule.

### Native resource behavior and validation

Native dimensions, index arithmetic, allocation sizes, and empty batches
are validated independently of either fusion flag. Prefill uses the existing
shared-memory kernels when their index/grid domains and device/function
shared-memory capacity permit. Legal extended shared memory is configured
to a fixed capacity after subtracting static shared memory, never to a
shape-dependent smaller limit. Function attributes are queried in the
current CUDA context; there is no process-global initialization cache.

During stream capture no shared-memory attribute is changed. An unwarmed
extended-memory case uses the existing direct kernel; already configured
cases can use the shared kernel. Channels-last layouts and shapes beyond
the shared kernel's capacity or index/grid domain also use the direct path.
That path can be slower for channels-first long sequences. Unexpected CUDA
query, configuration, and launch errors are propagated, not cleared and
retried as successful fallback.

Focused tests check exact history state, native dispatch boundaries,
source/fused/direct CUDA placement, and finite-input numerical errors against
an independent double-precision mathematical reference with dtype-aware
forward-error budgets. Qualification is scoped to the complete matched
region with identical inputs and history, including both outputs. Such local
bounds are not whole-model error bounds or a guarantee of long-generation
accuracy. Any separate model-level qualification requires independently
selected criteria, not thresholds fitted to observed mismatches. Finite
outputs or matching greedy tokens alone are insufficient.

```sh
./onnxruntime_test_all --gtest_filter='CUDARuntimeFusionHostTest.*:PatternRewriterTests.*:PatternComponentTests.*:PatternCapabilityTests.*'
./onnxruntime_provider_test --gtest_filter='CudaRuntimeFusionTest.*:CudaRuntimeFusionProbe.ZeroBatch:CausalConvWithStateTest.*-*LocalAuthentic*'
```
