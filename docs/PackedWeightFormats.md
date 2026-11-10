# Packed-weight format contract

This contract defines the packed-format attributes of version 1 of the `com.microsoft`
[GatherPacked](./ContribOperators.md#com.microsoft.GatherPacked),
[MatMulPacked](./ContribOperators.md#com.microsoft.MatMulPacked), and
[MoEPacked](./ContribOperators.md#com.microsoft.MoEPacked) operators.

## Stable identifiers and upstream layout revision

The IDs, block layouts, quantization arithmetic, and codebooks below are stable ORT model-format meanings,
not a request to interpret whichever ggml enum or decoder happens to be current. ORT updates must preserve
these meanings for existing models. An incompatible layout, codebook, or decoding change requires a new
format ID or operator version; an upstream enum or layout change must not silently reinterpret an existing
ORT model.

The reference layouts are the ggml implementation included in
[llama.cpp commit `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`](https://github.com/ggml-org/llama.cpp/tree/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc):

- [ggml.h](https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/ggml/include/ggml.h)
  defines the corresponding type numbers.
- [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/ggml/src/ggml-common.h)
  defines the packed block fields and IQ lookup tables, with `QK_K = 256`.
- [ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/ggml/src/ggml-quants.c)
  defines the bit assignments, sign decoding, scale arithmetic, and logical element order.
- [gguf-py/gguf/quants.py](https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/gguf-py/gguf/quants.py)
  provides the independent decoder used to verify ORT's fixed known-answer tests.

Only the IDs in this table are supported, subject to the per-operator restriction for NVFP4 below. All other values, including unlisted
IDs in the range `0..99`, negative IDs, and IDs `>=100`, are unsupported and rejected. A known upstream
enum value does not imply ORT support.

In the field descriptions, `f16` is IEEE binary16, `f32` is IEEE binary32, `bf16` is the upper 16 bits
of an IEEE binary32 representation, and `u8`, `i8`, and `u16` are unsigned or signed integers of the
stated width. Arrays are in stored field order, without padding.

| ID | Name | Elements per block | Bytes per block | Stored block fields |
|---|---|---:|---:|---|
| 0 | F32 | 1 | 4 | One `f32` value |
| 1 | F16 | 1 | 2 | One `f16` value |
| 2 | Q4_0 | 32 | 18 | `d: f16`, `qs: u8[16]` |
| 6 | Q5_0 | 32 | 22 | `d: f16`, `qh: u8[4]`, `qs: u8[16]` |
| 8 | Q8_0 | 32 | 34 | `d: f16`, `qs: i8[32]` |
| 13 | Q5_K | 256 | 176 | `d: f16`, `dmin: f16`, `scales: u8[12]`, `qh: u8[32]`, `qs: u8[128]` |
| 14 | Q6_K | 256 | 210 | `ql: u8[128]`, `qh: u8[64]`, `scales: i8[16]`, `d: f16` |
| 16 | IQ2_XXS | 256 | 66 | `d: f16`, `qs: u16[32]` (codebook indices, signs, and scales) |
| 17 | IQ2_XS | 256 | 74 | `d: f16`, `qs: u16[32]` (codebook indices and signs), `scales: u8[8]` |
| 18 | IQ3_XXS | 256 | 98 | `d: f16`, `qs: u8[96]` (64 index bytes followed by 32 sign/scale bytes) |
| 20 | IQ4_NL | 32 | 18 | `d: f16`, `qs: u8[16]` |
| 22 | IQ2_S | 256 | 82 | `d: f16`, `qs: u8[64]` (32 index bytes followed by 32 sign bytes), `qh: u8[8]`, `scales: u8[8]` |
| 23 | IQ4_XS | 256 | 136 | `d: f16`, `scales_h: u16`, `scales_l: u8[4]`, `qs: u8[128]` |
| 30 | BF16 | 1 | 2 | One `bf16` value |
| 40 | NVFP4 | 64 | 36 | `d: u8[4]` (UE4M3 scale per 16 elements), `qs: u8[32]` (E2M1 codes; each group's elements 0-7 use low nibbles, 8-15 high nibbles) |
| 42 | Q2_0 | 64 | 18 | `d: f16`, `qs: u8[16]`; consecutive two-bit codes `0,1,2,3` decode to `d * (-1,0,1,2)` |

NVFP4 is supported by CUDA MatMulPacked only; GatherPacked and MoEPacked reject ID 40.
Its `K` must be a positive multiple of 64. Scale bytes must have bit 7 clear (values
`0x00..0x7f`); bytes with bit 7 set are outside this contract. For scale byte `s`,
`0x00` and `0x7f` decode to zero. Otherwise, with `e = s >> 3` and `m = s & 7`,
the scale is `m * 2^-9` when `e == 0`, and `(1 + m/8) * 2^(e-7)` otherwise.
Thus `0x01` is `2^-9` and `0x7e` is 448. The code nibble indexes the doubled E2M1
table `[0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]`; the decoded weight is
`table[code] * 0.5 * scale`. Blocks concatenate four 16-element groups.

## Byte order and tensor layout

Every multi-byte scalar and packed integer field is **little-endian**, including scales and sign/scale
words assembled from byte arrays. Sub-byte fields use the bit positions and logical element order of
the pinned upstream decoder; for example, Q4_0 stores the first 16 values in the low nibbles and the next
16 values in the high nibbles, not adjacent low/high-nibble pairs.

Inputs are contiguous, row-major ONNX `uint8` tensors containing only packed block bytes. They do not
contain GGUF headers, tensor metadata, inter-tensor alignment gaps, or per-row padding. For a logical
row of `K` elements:

```text
K % block_elements == 0
row_bytes = (K / block_elements) * block_bytes
```

Each row is the concatenation of whole blocks in logical element order. GatherPacked and MatMulPacked
use `[vocab_size, row_bytes]` and `[N, row_bytes]`, respectively. MoEPacked uses
`[num_experts, inter_size, gate_row_bytes]`, `[num_experts, inter_size, up_row_bytes]`, and
`[num_experts, hidden_size, down_row_bytes]`; gate and up rows have `hidden_size` logical elements, and
down rows have `inter_size` logical elements. Its three format IDs may differ.

GatherPacked requires a positive `K` containing whole blocks. MatMulPacked additionally requires `K`
to be a positive multiple of 32 on CUDA. MoEPacked requires positive `hidden_size` and `inter_size`
that are multiples of 32 and whole blocks for their respective weight formats. CUDA execution
requires device-resident packed weights; host-backed external initializers are copied to device memory.

## CUDA MatMulPacked weight rounding

The packed layout is independent of the activation dtype and dispatch path. By default, every
complete decoded weight is rounded to the activation dtype before multiplication, for both decode
and prefill. Scale factors are combined in FP32 before that rounding, not rounded separately.
This also applies to FP16 prefill when its scratch buffers store the rounded weights in FP32.
Activation-dtype overflow and underflow apply to this rounding; for example, weights beyond the
finite FP16 range can overflow even if an unrounded FP32 dot product would have been finite.

The CUDA session option `ep.cuda.matmul_packed_allow_unrounded_gemv` controls an optional GEMV
precision relaxation. It accepts only the strings `"0"` and `"1"` and is resolved when the kernel
is created. Here, `M` is the product of the activation tensor's leading dimensions (one for a vector).

| Session option | Decode/GEMV, `M <= 8` | Prefill/GEMM, `M > 8` |
|---|---|---|
| Unset or `"0"` (default) | Activation-dtype-rounded weights | Activation-dtype-rounded weights |
| `"1"` | Eligible kernels may use unrounded FP32-decoded weights | Activation-dtype-rounded weights |

Setting `"1"` permits skipping rounding; it does not force a particular kernel or guarantee
unrounded weights. Tensor-core paths may still use activation-dtype-rounded weights.

For NVFP4 only, the optional scalar FP32 `global_scale` multiplies the FP32
accumulated result before the final activation-dtype cast, on every dispatch path;
it is not folded into the per-weight rounding. Omitting it is equivalent to 1.

All paths accumulate and reduce in FP32 and return the activation dtype. Consistent operand rounding
does not imply bitwise-identical results across kernels: their accumulation and reduction orders
can still differ.
