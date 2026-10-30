# CUDA HyperConnectionLowRankMix decode tuning

The CUDA kernel has two implementations. The operator schema, output layout, and
accumulation semantics are identical for both.

* **Fused path** (two kernels, `HcDownKernel` + `HcUpPreKernel`): 16-bit types
  (FP16/BF16), compute capability 8.0+, `rows <= 8`, `num_branches` in
  {1, 2, 4, 8}, `hidden % 8 == 0`, `hidden <= 4096`, `rank % 8 == 0`,
  `8 <= rank <= 512`, and 16-byte aligned tensors. The row group handled by one
  CTA depends on the row count (down rows x up rows):

  | rows | row groups |
  | ---- | ---------- |
  | 1    | 1 x 1      |
  | 2..7 | 2 x 2 (this includes four-row inputs) |
  | 8    | 4 x 2      |

  The 4 x 2 configuration is the speculative-decode shape for 8 rows
  (`batch * speculative tokens`). The fused path has no dedicated four-row
  configuration, and the row grouping is not configurable.
* **Unfused path** (any other shape, any type): branch-wise RMS norm, cuBLAS GEMMs
  with FP32 accumulation, and elementwise kernels. It supports an arbitrary
  `num_branches`, any row count, and unaligned tensors. 16-bit inputs with
  `hidden % 8 == 0`, `hidden <= 4096` and 16-byte aligned tensors use vectorized
  norm and pre-mix kernels. The inject GEMV and post-gate are fused in one
  deterministic kernel: a four-row kernel for 16-bit `num_branches == 4` with an
  8-byte aligned `inject_weight`, otherwise a one-row kernel tiled over groups of
  eight branches (so `num_branches > 8` is supported).

## Environment variables

Read once per process, so set it before starting it.

* `ORT_HYPER_CONNECTION_PDL` (default `1`): controls Hopper programmatic dependent
  launch (`cudaLaunchAttributeProgrammaticStreamSerialization`) between `HcDownKernel`
  and `HcUpPreKernel` of the fused path. It applies on `SM90+` with CUDA 12.3+, when the
  weight inputs (`norm_scale`, `down_weight`, `up_weight`, `inject_weight`) are
  constant initializers; `0` forces plain launches.

The rank-320 decode test (`HyperConnectionLowRankMixTest.FusedDecodeShapeRank320`)
covers 1, 2, 3, 4, 7 and 8 rows.

## Benchmark

rc3 model, INT4, H200, DFlash2 S=8 decode, ms per cycle at 2K / 8K / 32K context
(5 repetitions, noise about +-0.03 ms):

| configuration                      | 2K    | 8K    | 32K   |
| ---------------------------------- | ----- | ----- | ----- |
| default (4 x 2 rows, PDL on)       | 14.93 | 15.07 | 15.30 |
| 2 x 2 rows for 8 rows              | 15.00 | 15.11 | 15.38 |
| `ORT_HYPER_CONNECTION_PDL=0`       | 15.09 | 15.25 | 15.48 |

The PDL comparison pass measured 14.92 / 15.10 / 15.33 ms with PDL against
15.09 / 15.25 / 15.48 ms without, so PDL gives about 1%. The 2 x 2 variant for 8 rows
was 0.04-0.08 ms slower, so the row grouping is fixed at 4 x 2 and is no longer
configurable.
