# Numerical semantics

**Owns:** Arithmetic accuracy, representation conversions, quantization, masks,
and discrete decisions. Address bounds belong in dispatch-and-wgsl.
**Update when:** Arithmetic helpers, supported formats, precision controls, or
operator mathematical contracts change.

- Trace storage, accumulator, partial-result storage, divisor, and final cast
  separately. f16 sums overflow above 65504; use f32 reduction/dot accumulation
  unless the precision contract specifies otherwise. Widening an accumulator
  does not prevent f16 Split-K partials or divisors from overflowing/rounding.
  `enableMatmulFp32Accumulation` affects MatMul, Gemm, and MatMulNBits; follow
  shared helpers to Conv and all other callers and generated type aliases.
- Signed WGSL `/` and `%` truncate toward zero. Preserve ONNX floor semantics
  (`Mod` with `fmod=0`), negative-index wrapping, and full-width integer math.
  For int64 encoded as two u32 words, prove carries, signs, comparisons, and
  quotients; a low-word conversion or saturated index is not general int64 math.
- Distinguish an absent attribute from an explicit zero and an absent optional
  tensor from a present input slot. Keep scale, epsilon, axis, mask, and cache
  semantics for each operator/domain/opset.
- WGSL `min`, `max`, and `clamp` do not guarantee NaN propagation; `round` uses
  half-to-even. Implement required NaN/infinity, saturation, and tie behavior.
  Quantization thresholds must use the same arithmetic domain as the quantized
  value. Test below/at/above even and odd ties, non-power-of-two scales,
  subnormals, signed extremes, and bit-level encodings when applicable.
- Precision before TopK/ArgMax/routing can change indices despite a small final
  numeric error. Distinguish finite negative maximum from negative infinity;
  padding/masked candidates must lose before applying stable-index tie rules.
- Normalization needs constant, nearly constant, large-offset, cancellation,
  and permuted rows, non-unit Scale, and optional Mean/InvStd outputs. Raw second
  moments and premature mean rounding can fail even in f32. Preserve full
  LayerNorm versus RMSNorm math and independently exported Q/K epsilons.
- Force scalar, packed, subgroup-matrix, and Split-K variants where eligible.
  Choose reductions long enough to expose the precision failure; do not broaden
  tolerances without an error analysis. Scope performance tradeoffs to evidence.

Review evidence: [#33057 partial storage](https://github.com/microsoft/onnxruntime/pull/33057#discussion_r4163219361),
[shared callers](https://github.com/microsoft/onnxruntime/pull/33057#discussion_r4169930321),
[#32528 full-width indices](https://github.com/microsoft/onnxruntime/pull/32528#discussion_r4168826629),
[precision before selection](https://github.com/microsoft/onnxruntime/pull/32528#discussion_r4168824365),
and [#33074 quantization domains](https://github.com/microsoft/onnxruntime/pull/33074#discussion_r4198720082).
These are regression prompts; read replies and the affected revision before
claiming a current defect.
