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

`ProjectedSigmoidGateMulKernel`
([gated_add_impl.cu](../../../onnxruntime/contrib_ops/cuda/bert/gated_add_impl.cu)),
which precedes `HcDownKernel` in the Qwen3.8 decode graph, issues
`griddepcontrol.launch_dependents` at its start on `SM90+`. When `HcDownKernel`
is launched with PDL, its prologue (constant-weight prefetch) then overlaps with
the gate-mul kernel, while its reads of the gate-mul output still wait for
`griddepcontrol.wait`. The trigger has no effect for plain launches, changes no
result and has no switch. Together with the same early trigger in the fpA_intB GEMV
kernels (MatMulNBits) it shortened a Qwen3.8-Flash-Next `S = 8` decode cycle by 1.4-1.7% in a
preliminary end-to-end A/B on an H200 (one round, not alternated, no confidence interval).

`SigmoidF32` evaluates `expf(-|v|)` once and selects the numerator,
`e = expf(-|v|); (v >= 0 ? 1 : e) / (1 + e)`. The textbook two-branch form passes the
same argument to `expf` (`-v` for `v >= 0`, `v` otherwise), so the result is
bit-identical for all 2^32 `float` values (checked exhaustively in a standalone harness,
no fast-math). The branch itself was the cost: in the prefill pre-mix neighbouring
elements have different signs, so nearly every warp executed both sides. For hidden
2560, 4 branches and an 8192-token prefill chunk on an H200, `PreMixVec8Kernel` takes
82.7 µs instead of 112.1 µs per launch and `ScaledSiluKernel` 6.1 µs instead of 7.3 µs.
The TTFT shortened by 0.3-1.1% in a preliminary end-to-end A/B (median of 5 repetitions,
faster in 9 of 9 package and context cells), which is about the size of the roughly 1%
noise between repetitions, so the kernel times are the firmer evidence. The fused decode
kernels call the same function for at most eight rows, and decode time is unchanged within
the noise of 0.3%.

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
