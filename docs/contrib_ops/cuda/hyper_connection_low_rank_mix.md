# CUDA HyperConnectionLowRankMix decode tuning

The CUDA kernel has two implementations. The operator schema, output layout, and
accumulation semantics are identical for both.

* **Fused path** (two kernels, `HcDownKernel` + `HcUpPreKernel`): 16-bit types
  (FP16/BF16), compute capability 8.0+, `rows <= 8`, `num_branches` in
  {1, 2, 4, 8}, `hidden % 8 == 0`, `hidden <= 4096`, `rank % 8 == 0`,
  `8 <= rank <= 512`, and 16-byte aligned tensors. The row group handled by one
  CTA depends on the row count (down rows x up rows):

  | rows        | row groups | selected by                                   |
  | ----------- | ---------- | --------------------------------------------- |
  | 1           | 1 x 1      | always                                        |
  | 2 .. 7      | 2 x 2      | always (this includes four-row inputs)        |
  | 8 (default) | 4 x 2      | default (`ORT_HC_S8_DECODE_ROWS=4`)           |
  | 8           | 2 x 2      | `ORT_HC_S8_DECODE_ROWS=2`                     |

  The 4 x 2 configuration is the speculative-decode default for 8 rows
  (`batch * speculative tokens`). The fused path has no dedicated four-row
  configuration.
* **Unfused path** (any other shape, any type): branch-wise RMS norm, cuBLAS GEMMs
  with FP32 accumulation, and elementwise kernels. It supports an arbitrary
  `num_branches`, any row count, and unaligned tensors. 16-bit inputs with
  `hidden % 8 == 0`, `hidden <= 4096` and 16-byte aligned tensors use vectorized
  norm and pre-mix kernels. The inject GEMV and post-gate are fused in one
  deterministic kernel: a four-row kernel for 16-bit `num_branches == 4` with an
  8-byte aligned `inject_weight`, otherwise a one-row kernel tiled over groups of
  eight branches (so `num_branches > 8` is supported).

## Environment variables

Read once per process, so set them before starting it.

* `ORT_HC_S8_DECODE_ROWS` (experimental, default `4`, allowed values `2` and `4`;
  other values fail the fused launch): applies only to the fused path with
  `rows == 8`. `4` selects the 4 x 2 kernels and `2` selects 2 x 2. It has no effect
  for other row counts.
* `ORT_HYPER_CONNECTION_PDL` (experimental, default `1`): controls Hopper programmatic dependent
  launch (`cudaLaunchAttributeProgrammaticStreamSerialization`) between `HcDownKernel`
  and `HcUpPreKernel`. It applies on `SM90+` with CUDA 12.3+, when the weight inputs
  (`norm_scale`, `down_weight`, `up_weight`, `inject_weight`) are constant
  initializers; `0` forces plain launches.

The rank-320 decode test (`HyperConnectionLowRankMixTest.FusedDecodeShapeRank320`)
covers 1, 2, 3, 4, 7 and 8 rows. Run it in separate processes with
`ORT_HC_S8_DECODE_ROWS=4` (default) and `ORT_HC_S8_DECODE_ROWS=2` to exercise both
eight-row configurations.
