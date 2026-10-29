# Provider-specific Log1p fusion

Export portable ONNX `Log(Add(x, one))` (either Add input order), not a new ONNX
operator. With extended optimizations (`ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL`),
ORT replaces eligible, already provider-assigned patterns with `com.microsoft.Log1p`.
Disabled/basic optimization leaves Add and Log separate. `Log1pFusion` can also be
disabled by name using ORT's transformer-disable configuration.

The Add and Log must belong to the same CUDA or native WebGPU provider. The
constant must be an immutable initializer of exactly one, with the input's dtype,
and scalar or all-singleton dimensions of rank no greater than the input.
Nonsingleton constants, rank-expanding broadcasts, shared Add consumers, and Add
graph outputs are conservatively left unchanged. CPU, other EPs, and deprecated
JSEP (`JsExecutionProvider`) are not rewritten.

The kernel receives **x**, not the rounded Add output: FLOAT `x=1e-8` gives a
nonzero result close to `1e-8`, whereas a separate FLOAT Add produces exactly one
and its Log is zero. This optimization intentionally changes rounding relative to
the unfused graph. It does not combine preceding operations: for example, an
`Exp -> Add -> Log` graph retains the separate Exp and its dtype rounding.

CUDA uses native `log1pf` for FLOAT, and float opmath followed by narrowing for
FLOAT16/BFLOAT16. DOUBLE uses native `log1p`. Native WebGPU supports FLOAT and,
only on adapters with `shader-f16`, FLOAT16 (f32 opmath then narrowing); it does
not support DOUBLE or BFLOAT16. WGSL has no native log1p: for absolute inputs at
most 0.5 the shader evaluates the odd atanh series through degree 17 at
`s=x/(2+x)`; elsewhere it uses `log(1+x)`. This avoids tiny-input cancellation,
but is not CUDA/Torch bit-identical. Inputs with absolute value at most `2^-25`
are returned directly, which is correctly rounded in f32 and avoids an
underflowing intermediate for minimum-normal FLOAT inputs. This protects normal
inputs; it does not promise preservation of backend-flushed subnormal inputs.
WebGPU transcendental accuracy, subnormal
flushing, signed-zero preservation and nonfinite/domain behavior remain subject
to WGSL and backend guarantees; applications must not rely on portable IEEE
NaN/infinity results outside the finite domain `x > -1`.

The CUDA fused kernel includes BFLOAT16, but the existing standard CUDA Log
kernel does not. Consequently, a BFLOAT16 Add/Log graph is not directly eligible
for this post-partition fusion; kernel availability does not imply that every
source pattern can be assigned to that provider.

For ORT Web, `onnxruntime-web/webgpu` and `onnxruntime-web/jspi` use the native
provider. The default/JSEP artifacts resolve operators separately and do not
include this new fusion/kernel; new JSEP operators are excluded by
[JSEP's maintenance policy](JSEP_Deprecation.md).
