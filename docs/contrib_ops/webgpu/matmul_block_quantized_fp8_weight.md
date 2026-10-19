# WebGPU block-quantized FP8 dense MatMul

`com.microsoft::MatMulBlockQuantizedFp8Weight` supports FP16 activations and
row-major E4M3FN weights with FP32 block scales shaped
`[N, ceil(K / block_size)]`. It supports optional activation quantize/dequantize,
bias, rank-one and batched inputs, partial blocks, and non-word-aligned weight rows.
BF16 activation storage is not supported by this WebGPU kernel.

When activation scaling is supplied, quantization uses saturating E4M3FN
round-to-nearest, ties-to-even semantics. NaN activations remain NaN rather than
being replaced by the maximum finite value. Weight decoding preserves finite
E4M3FN subnormals before multiplication.

Prefill can use 16x16x16 FP16 subgroup matrices with FP32 accumulation when the
adapter exposes the required configuration, subgroup-size requirements, and the
instance exposes the WGSL `subgroup_id` language feature. Other configurations
use the scalar FP32-reduction path. This is FP8 storage with FP16 widening, not
native FP8 matrix multiplication.

Shared shader generation includes `@builtin(subgroup_id)` only for programs
that explicitly request it. Both dense FP8 and routed block-FP8 QMoE matrix
programs check the language feature before selecting that path. Ordinary
operators and scalar fallbacks do not acquire this language-feature requirement.

The focused `MatMulBlockQuantizedFp8WeightTest.*` tests disable CPU EP fallback.
Matrix-specific cases check capabilities and skip explicitly on unsupported
adapters; scalar cases remain separate.
