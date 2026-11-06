# MatMulPacked

`com.microsoft.MatMulPacked` version 1 computes `Y = A @ dequantize(B).T`
from native GGUF tensor block bytes. It does not parse a GGUF file.

The initial codecs are `quant_type="iq2_xxs"`, `"q8_0"` and `"nvfp4"`. Required integer attributes `K`
and `N` give the logical weight dimensions. BF16 input `A` has shape
`[..., K]`; UINT8 input `B` has shape `[N, (K / 256) * 66]` for IQ2_XXS
or `[N, (K / 32) * 34]` for Q8_0, or `[N, (K / 64) * 36]` for NVFP4. The output is
BF16 with shape `[..., N]`.

Each 256-weight IQ2_XXS block contains the original little-endian FP16 scale
in its first two bytes and eight consecutive 8-byte records in the remaining
64 bytes. There is no padding between blocks. Each record contains four grid
indices followed by a little-endian word holding four 7-bit sign indices and
a 4-bit local multiplier. Scales are not separate inputs.

The kernel forms `d * (0.5 + local_multiplier) * 0.25` in FP32, multiplies
the signed grid value by that scale, and rounds the complete operand to BF16.
MMA accumulation and inter-warp reduction use FP32, followed by BF16 output
rounding. Rounding the scale to BF16 before multiplying the grid is not
equivalent.

Each 32-weight Q8_0 block contains a little-endian FP16 scale followed by
32 signed INT8 weights. The kernel multiplies each signed value by the FP16
scale in FP32, rounds the weight to BF16, and uses the same FP32 MMA and
inter-warp reduction. This is not integer-dot-then-scale arithmetic.
Converting a non-GGUF BF16-scale checkpoint to FP16 requires a losslessness
check; the codec does not reinterpret BF16 scale bits as FP16.

Each 64-weight NVFP4 block contains four unsigned E4M3 scale bytes followed
by 32 E2M1 bytes. Within each 16-weight group, a byte pairs element `j` in
the low nibble with element `j+8` in the high nibble. These bytes remain
unmodified in storage; the kernel arranges the fragments for its MMA/GEMV
engine. Canonical scale bytes are 0 through 127; GGUF reserves 127 for zero.
The separate FP32 `global_scale` input is required as a scalar or one-element
vector for NVFP4 and must be absent for the other codecs.

NVFP4 preserves the source's distinct arithmetic: one-row GEMV uses FP16
paired partial sums with activations pre-scaled by `2^-6`; multiple rows
use rescaled BF16 operands and FP32 MMA/reduction. The FP32 global multiplier
is applied after accumulation. Very large activations can overflow the GEMV
FP16 partials. This is not a promise of identical one-row/multirow arithmetic,
or of equivalence to global scaling before BF16 weight rounding.

The initial CUDA implementation requires SM80+, positive `K` divisible by
256 for IQ2_XXS or 64 for Q8_0, or `K=4096/8192` for NVFP4, positive `N`,
and 1 through 8 flattened activation rows. It preserves
leading activation dimensions. It has no CPU kernel, fused bias or
general-prefill implementation. Unsupported codecs and shapes fail explicitly.
UINT8 bindings need only byte alignment; BF16 bindings need two-byte alignment.

The private MMA helpers and reduction topology originate from the qualified
projection implementation at commit
`491dc043952066267b2212ba979f0325da318573`. Native block addressing replaces
the separate record/scale addressing without changing operand reconstruction.
The IQ2 codebook provenance and MIT notice are retained in
`packed_projection_impl.cuh` and `ThirdPartyNotices.txt`.

Tests cover native block decoding with an independent scalar fixture,
multiple contributing superblocks, leading dimensions, dispatch boundaries,
invalid payloads/codecs, runtime row limits and offset bindings. Kernel
layout comparisons are not by themselves a full-model or performance claim.
