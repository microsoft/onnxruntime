# RMSNormRotary (CUDA)

`com.microsoft::RMSNormRotary` version 1 combines ONNX-arithmetic BF16 RMS
normalization with non-interleaved, optionally partial rotary embedding.
It replaces `BranchwiseRMSNorm(use_onnx_arithmetic=1)` followed by a flattening
reshape and `RotaryEmbedding`. It does not change projection weights.

## Contract

| Input | Type | Shape |
| --- | --- | --- |
| `X` | BF16 | `[B,S,heads,stride]` |
| `scale` | FP32 | `[width]` |
| `position_ids` | INT64 | One base position, or `[output_batch,output_sequence]` |
| `cosine`, `sine` | BF16 | Both `[cache_rows,rotary_embedding_dim/2]` |
| `prefill_numerics` | Optional CPU INT32 | One value: 0 (decode) or 1 (prefill) |

| Attribute | Default | Meaning |
| --- | --- | --- |
| `epsilon` | `1e-6` | Finite, nonnegative normalization epsilon |
| `input_start` | `0` | First feature of the contiguous normalized slice |
| `input_width` | `0` | Slice width; zero selects the full input last dimension |
| `rotary_embedding_dim` | Required | Positive even value no larger than width |
| `use_fp32_arithmetic` | `1` | Decode rotary arithmetic: FP32 (1) or input-type (0) |
| `flatten_batch` | `0` | One merges B and S before applying rotary position semantics |

Output `Y` is BF16 `[B,S,heads*width]`, or `[1,B*S,heads*width]` when
`flatten_batch=1`. Optional `normalized_fp32` is `[B,S,heads,width]` and contains
the BF16-rounded normalized values widened to FP32 **before rotation**.

Normalization squares and reduces FP32 values, adds epsilon, computes `sqrt`,
normalizes, multiplies the FP32 scale, and rounds to BF16. Decode divides by
the RMS; prefill multiplies by its FP32 reciprocal, matching the existing
`BranchwiseRMSNorm` numerical-mode control. The scale is already the effective
weight, including any exporter-side weight offset.

Rotation preserves the BF16 normalization boundary. FP32 decode rotation uses
separately rounded FP32 multiplies/addition before the BF16 store. Prefill uses
the existing native BF16 rotary intermediate arithmetic. The nonrotary tail is
copied unchanged. Cache bounds and scalar-base versus explicit-position
behavior match the existing rotary kernel; no position tensor is read on CPU.

## Dispatch and correctness

Width 256 with more than one normalized row uses one 64-thread block per head
row for FP32 decode rotary or prefill. It retains the original column
reduction's four strided elements per thread, two warp shuffle reductions, and
final two-warp addition. Combining reduction, normalization and rotation
removes the intermediate norm tensors and separate epilogue/rotary launches.

Other widths, a single normalized row, and input-type decode rotation reuse
the original standard normalization and rotary launchers. This is deliberate:
single-row normalization selects a different reduction, and input-type decode
rotation has a distinct mapped-type rounding contract. No numerical tolerance
is widened to enable fusion.

`FusedTensorOpsTest.RMSNormRotaryExactUnfusedParity` compares both outputs
bit-exactly with the original CUDA operators across packed slices, multiple
batches, scalar/explicit positions at cache boundaries, flattened batches,
partial rotary, empty/one/two/seven-token sequences, both numerical modes,
and optimized/fallback widths. Invalid shape, scale, slice, control and
attribute contracts are tested separately.
