# Source-exact packed BF16 projections

These opt-in CUDA profiles of existing `com.microsoft` version-1 operators retain the frozen packed
projection arithmetic. They require SM80 or later, BF16 activations/output,
positive N/K, and **1 through 8 flattened activation rows**. Leading dimensions
are preserved. They do not provide general prefill, fused bias, or CPU kernels.
They do not change the existing operators' default dispatch, GGML or NVFP4A16 primitives.

| Operator and profile | Packed B | Separate scales | Arithmetic |
|---|---|---|---|
| `MatMulBlockQuantizedFp4Weight`, `gemv_kernel="fixed"` | UINT8 `[N,K/2]`, low-nibble-first E2M1 | UINT8 E4M3 `[N,K/16]`, one FP32 global multiplier | Row one uses scaled FP16-pair FMA partial sums; rows 2–8 use BF16 MMA, FP32 accumulation and block reduction. Apply global multiplier before output rounding. |
| `MatMulNBits`, `gemv_kernel="fixed"`, `bits=8`, `block_size=32` | UINT8 `[N,K/32,32]`, **byte minus 128** | FP16/BF16 `fixed_scales` `[N,K/32]` | FP16-scale inputs always reconstruct each full weight in FP32 before BF16 rounding/MMA. BF16-scale row one instead scales FP32 block dot products against unscaled integer weights; rows 2–8 use reconstructed BF16 MMA operands. |
| `MatMulNBits`, `gemv_kernel="fixed_iq2_xxs"`, `bits=2`, `block_size=256` | UINT8 `[N,K/256,64]`, eight aligned records per superblock; each record contains four grid bytes followed by a little-endian sign/local-scale word | Literal FP16 `fixed_scales` `[N,K/256]` | Form FP32 block scale `d*(0.5+s)*0.25`; round the complete signed grid-times-scale operand to BF16, then MMA and block reduction. Do **not** round the block scale first. |

Both `MatMulNBits` profiles require `K` and `N` attributes, omit inputs 2 through 5
(`scales`, `zero_points`, `g_idx`, `bias`), and supply independent-type
`fixed_scales` at input 6. Q8 BF16 scales are genuine BF16 values and must not be
reinterpreted as FP16. IQ2 rejects BF16 scale tensors: an exporter migrating a
raw-bit carrier must expose the same bytes as literal FP16, not numerically cast
them. Reshaping aligned IQ2 records into superblocks does not change payload bytes.

NVFP4 requires K divisible by 1024; row one additionally requires K=4096 or
8192. Q8 requires K divisible by 64; its BF16-scale row-one path requires
K<=6144. IQ2XXS requires K divisible by 256.

All MMA paths fix the source's default reduction topology: MT=2/KW=4 for
N>=8192, MT=1/KW=16 otherwise. The private source environment-variable tuning
overrides are deliberately not part of these operator contracts.

## Why explicit profiles?

`MatMulNVFP4A16` uses rounded local weights and FP32 accumulation for all row
counts; the frozen NVFP4 row-one FP16-pair partial sums have a different
rounding contract. `MatMulGGMLQ8_0` uses interleaved signed 34-byte records,
not offset-binary bytes with separate scales. `MatMulGGMLIQ2XXS` uses
interleaved 66-byte records, not aligned 8-byte records with separate raw
FP16 scales. Layout similarity or a checkpoint's source label is not an
equivalence proof. The explicit profiles reuse the existing public operators
without silently changing their default accumulation or codec contracts.

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
K=4096/8192, Q8 BF16 row-one IT=2/4/6 through K=6144, and literal IQ2 FP16
scales with explicit rejection of BF16 scale tensors. The NVFP4 row-one cancellation case places
`[16384, 1, -16384]` in the same low FP16 accumulator (paired with activation
indices 4/5/6); its required output is 0, not the FP32 sum 1. Q8 has a separate
row-one integer-dot cancellation case. Unsupported >8-row prefill is checked
with static graph shapes and omitted shapes (runtime guard).
These fixtures are collected by the existing contrib test glob; execution
requires CUDA SM80 or later. CUDA sources are collected by the existing recursive
provider glob.

Lower-level component validation is not a full ORT/GenAI integration or
full-model memory/performance qualification. In particular, supporting the
frozen decode/verify projections does not implement a general-prefill path.
