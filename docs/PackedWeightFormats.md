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

Only the IDs in this table are supported by these three operators. All other values, including unlisted
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
| 42 | Q2_0 | 64 | 18 | `d: f16`, `qs: u8[16]`; consecutive two-bit codes `0,1,2,3` decode to `d * (-1,0,1,2)` |

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

The packed layout is independent of the activation dtype and dispatch path. The current CUDA paths
dequantize and round weights as follows, where `M` is the product of the activation tensor's leading
dimensions (one for a vector):

| Activation dtype | Dispatch | Weights used in multiplication |
|---|---|---|
| FP16 | Decode, `M <= 8` | FP32-decoded weights |
| FP16 | Prefill, `M > 8` | FP32-decoded weights, except F16-packed weights remain FP16 |
| BF16 | Decode, `M <= 8` and `N < 65536` | FP32-decoded weights |
| BF16 | Decode, `M <= 8` and `N >= 65536` | BF16-rounded decoded weights, including the SIMT fallback |
| BF16 | Prefill, `M > 8` | BF16-rounded decoded weights |

All paths accumulate and reduce in FP32 and return the activation dtype. Thus changing `M` or `N`
across a dispatch boundary can change a BF16 result even when each activation row is identical.
For example, with `K = 32`, activations of ones, and F32-packed weights
`[1.00390625, -1, 0, ...]`, small-`N` BF16 decode returns `0.00390625`, while BF16 prefill and
large-`N` BF16 decode return `0`. This is intentional weight rounding, not a different packed format.
