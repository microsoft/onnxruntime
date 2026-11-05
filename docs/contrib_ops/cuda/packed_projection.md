# Source-exact packed BF16 projections

These CUDA-only `com.microsoft` version-1 operators retain the frozen packed
projection arithmetic. They require SM80 or later, BF16 activations/output,
positive N/K, and **1 through 8 flattened activation rows**. Leading dimensions
are preserved. They do not provide general prefill, fused bias, or CPU kernels.
They do not change the existing GGML or NVFP4A16 primitives.

| Operator | Packed B | Separate scales | Arithmetic |
|---|---|---|---|
| `MatMulPackedNVFP4` | UINT8 `[N,K/2]`, low-nibble-first E2M1 | UINT8 E4M3 `[N,K/16]`, one FP32 global multiplier | Row one uses scaled FP16-pair FMA partial sums; rows 2–8 use BF16 MMA, FP32 accumulation and block reduction. Apply global multiplier before output rounding. |
| `MatMulPackedQ8` | UINT8 `[N,K/32,32]`, **byte minus 128** | FP16/BF16 `[N,K/32]` | FP16-scale inputs always reconstruct each full weight in FP32 before BF16 rounding/MMA. BF16-scale row one instead scales FP32 block dot products against unscaled integer weights; rows 2–8 use reconstructed BF16 MMA operands. |
| `MatMulPackedIQ2XXS` | UINT8 `[N,K/32,8]`, four grid bytes followed by a little-endian sign/local-scale word | Raw FP16 `[N,K/256]`; BF16 tensors may carry identical FP16 bits | Form FP32 block scale `d*(0.5+s)*0.25`; round the complete signed grid-times-scale operand to BF16, then MMA and block reduction. Do **not** round the block scale first. |

NVFP4 requires K divisible by 1024; row one additionally requires K=4096 or
8192. Q8 requires K divisible by 64; its BF16-scale row-one path requires
K<=6144. IQ2XXS requires K divisible by 256.

All MMA paths fix the source's default reduction topology: MT=2/KW=4 for
N>=8192, MT=1/KW=16 otherwise. The private source environment-variable tuning
overrides are deliberately not part of these operator contracts.

## Why separate operators?

`MatMulNVFP4A16` uses rounded local weights and FP32 accumulation for all row
counts; the frozen NVFP4 row-one FP16-pair partial sums have a different
rounding contract. `MatMulGGMLQ8_0` uses interleaved signed 34-byte records,
not offset-binary bytes with separate scales. `MatMulGGMLIQ2XXS` uses
interleaved 66-byte records, not aligned 8-byte records with separate raw
FP16 scales. Layout similarity or a checkpoint's source label is not an
equivalence proof.

## Port provenance

The dense kernels are mechanically transplanted from these frozen local source
snapshots; graph/ABI wrappers, tuning overrides and unrelated fused kernels are
not imported:

* `export-nemotron-pr-20261030/src/bolt_ops.cu`:
  SHA256 `1f587d9eacc356805c37d19e94e7aa4465dd50302576b2ca4b9579d7c5ae8ba5`
  (NVFP4 and BF16-scale Q8).
* `IQ2_XXS_cuda_bolt/compat/bolt_ops.cu`:
  SHA256 `a211ddf1c5a9b63d745ccdeb370f18c51b78b3cdf6f3bd51cb9cbf9c8619e91f`
  (FP16-scale Q8; its rank-compatible wrapper preserves leading dimensions).
* `IQ2_XXS_cuda_bolt/iq2-compat/bolt_ops.cu`:
  SHA256 `4a539c0e82b8ea48c2f3adc51dc324d0f792554bca2c46706b972ca5afc97345`
  (native IQ2 with complete-operand rounding). The unmodified upstream
  snapshot has different IQ2 rounding and must not substitute for this file.

The IQ2 codebook matches `iq2xxs_grid` in `ggml/src/ggml-common.h` at
ggml/llama.cpp commit `3de52ac38818978d6da8f386eda43a856e53714a`, with byte
magnitudes re-encoded as BF16 pairs. Its complete upstream MIT notice is retained
beside the table and in `ThirdPartyNotices.txt`.
`packed_projection_impl.cuh` exposes the packed load/dequantization/MMA/reduction
helpers for routed kernels without duplicating their arithmetic.

## Validation scope

Focused C++ operator fixtures cover nonunit scales, row counts 1/3/8, N=8192/8193
dispatch boundaries and column tails, multiple contributing K chunks, NVFP4
K=4096/8192, Q8 BF16 row-one IT=2/4/6 through K=6144, and both IQ2 FP16 and
raw-FP16-bits-in-BF16 scale inputs. The NVFP4 row-one cancellation case places
`[16384, 1, -16384]` in the same low FP16 accumulator (paired with activation
indices 4/5/6); its required output is 0, not the FP32 sum 1. Q8 has a separate
row-one integer-dot cancellation case. Unsupported >8-row prefill is checked
with static graph shapes (schema rejection) and omitted shapes (runtime guard).
These fixtures are collected by the existing contrib test glob; execution
requires CUDA SM80 or later. CUDA sources are collected by the existing recursive
provider glob.

Lower-level component validation is not a full ORT/GenAI integration or
full-model memory/performance qualification. In particular, supporting the
frozen decode/verify projections does not implement a general-prefill path.
