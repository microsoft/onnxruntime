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
They cover scalar, empty, odd, dynamic, and model-shaped inputs, FLOAT raw-bit
samples, and all BF16 bit patterns. Graph and profile assertions verify CUDA
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
