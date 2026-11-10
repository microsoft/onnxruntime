# MatMulNBits CPU: three-trial profiling report

- Runtime: `ORT Build Info: git-branch=tommasoadani/matmul_estimation_experiments, git-commit-id=fd4979a3de, fp8-kv-cache=1, build type=Release`
- Baseline: `/datadisks/disk10/tommasoadani/tmp/matmul-baseline-2026-11-09`
- INT4 weights; block size 32; no bias; implicit midpoint zero points.
- CPU only; one intra-op thread; sequential; graph optimizations disabled.
- Three independent trial pairs per configuration.
- Each process performs 100 warm-up runs followed by 1,000 or 2,000 runs.
- Event: `fp_ret_sse_avx_ops.all:u`; all accepted counters ran 100%.
- Per-trial value: `(count_2000 - count_1000) / 1000`.

## Scope and limitations

Measured values describe recurring process FP work, including runtime work. Independent-process subtraction reduces fixed startup/warm-up contributions but does not guarantee their cancellation. Estimates cover modeled operator components, not whole-op work. INT8 activation quantization and other excluded arithmetic can contribute to measured counts. Signed differences are not isolated operator accuracy percentages. Repetition assesses variability, not systematic hardware-counter bias.

## Reproducing the workflow

The original collection used inline commands. Reusable equivalents are now available:
[matrix generation](../../../../tools/python/generate_matmul_nbits_matrix.py),
[estimator validation](../../../../tools/python/run_matmul_nbits_experiments.py), and
[hardware collection/report generation](../../../../tools/python/profile_matmul_nbits.py).
See the [command sequence and requirements](../../../../tools/python/PythonTools.md#repeatable-matmulnbits-matrix-and-hardware-profiling).
The report command recomputes statistics from raw counters and reproduced all 84 rows of the saved CSV summary.
The reusable scripts preserve the method; they are not claimed to be the exact original executed driver.

## Accuracy request key

The **Accuracy request** column is the MatMulNBits `accuracy_level` attribute, not an error percentage.
It permits internal compute choices; it does not force a particular kernel or change the INT4 weight storage.
The **Mode** column records the effective compute mode observed during the estimator run.

| Accuracy request | Meaning |
|---:|---|
| 0 | Default/unset; no activation quantization or downcasting requested |
| 1 | FP32 accuracy |
| 2 | Permit FP16-level computation |
| 3 | Permit BF16-level computation |
| 4 | Permit INT8 activation quantization/computation |

In this CPU/build matrix, requests 0-3 selected `sqnbit_fp32`, including FP16 inputs at request 0.
Request 4 selected `sqnbit_int8` for both FP32 and FP16 inputs. Native FP16/BF16 compute modes were not observed.

## Follow-up audit: FP32 single-row AVX2 kernel (November 10)

For FP32 input `[1,1024]`, N=1024, accuracy request 0, INT4/block size 32, no bias and implicit midpoint zero points,
cycle sampling identified the actual single-row implementation:

```text
SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_avx2<false> [clone .isra.0]
```

The template argument means no explicit zero-point input; the implicit midpoint is still subtracted.
The clone suffix denotes compiler specialization. This establishes AVX2 kernel identity for the audited case,
not every case labeled `packed_qnbit` / `sqnbit_fp32`. The installed native libraries matched the saved baseline.
The cycle profile collected 4,833 samples; the kernel had 21.57% self cycle attribution. That percentage is
neither an FP-operation fraction nor a steady-state-only timing measurement.

The [single-row kernel](../../../../onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2.cpp) selects
`ComputeDotProducts_BlkLen32Plus_CompFp32_avx2<4, false, false>` for this aligned four-column/block-32 case.
Its midpoint subtraction uses integer `_mm256_sub_epi8`, followed by integer-to-float conversion, one FP scale
multiplication per weight and FP multiply-add accumulation. Integer subtraction and pure conversion are not
counted as FLOPs in this ledger.

The [FoldAccumulators helper](../../../../onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx_common.h)
uses three eight-lane FP additions and one four-lane addition: 28 FLOPs per four output columns, or 7N.
For this full-block, N-divisible-by-four, no-bias specialization:

| Component | Independent source-derived FP count |
|---|---:|
| Integer midpoint subtraction / pure conversion | 0 |
| Weight scaling, NK | 1,048,576 |
| Dot-product accumulation, 2NK | 2,097,152 |
| Final reductions, 7N | 7,168 |
| Kernel total, 3NK + 7N | **3,152,896** |
| Previously measured recurring process count | **3,152,898** |
| Process count minus source-derived kernel total | **+2** |
| Existing estimator subtotal, 4NK | **4,194,304** |

This explains the dominant discrepancy: the nominal estimator charges two FP reconstruction operations per weight,
but this specialization performs the zero-point subtraction as integer work, leaving one FP scale multiplication.
It also performs reductions excluded from the current three-component model. The two-operation residual is
unattributed; do not assume it is a particular runtime instruction without further evidence.

The sampled binary was annotated successfully, but the shared excerpt showed only the beginning of a function
containing multiple block-size branches. Full compiled block-32 control-flow/instruction accounting and SIMD/FMA
counter qualification remain follow-up checks. The close source-derived/process-count match is strong corroboration,
not a blanket instruction-exact or whole-op accuracy claim.

No estimator formula was changed by this audit. Do not generalize `3NK + 7N` to larger M, column/K tails, other
ISAs, explicit-zero-point variants or INT8 routes without validating their actual implementation.
The follow-up comparisons and held-out shapes below corroborate the bounded source model.
Full compiled block-32 instruction accounting remains open.
The raw local audit is at `/datadisks/disk10/tommasoadani/tmp/matmul-m1-audit-NecOJP/`
(`cycles.data`, `hot-symbols.txt`, `kernel-assembly.txt`, `runtime.json`, `case-estimates.json`).
These local paths are provenance references, not portable repository assets.
See the [repeatable audit sequence](../testing_strategy.md#repeatable-fp32-m1-kernel-audit).

### Additional aligned M=1 comparisons

The same source-derived formula also matches the two smaller geometries from the original matrix:

| K | N | Source-derived kernel FP ops, 3NK + 7N | Measured recurring process FP ops | Residual |
|---:|---:|---:|---:|---:|
| 64 | 32 | 6,368 | 6,370 | +2 |
| 128 | 64 | 25,024 | 25,026 | +2 |
| 1,024 | 1,024 | 3,152,896 | 3,152,898 | +2 |

Sampling confirmed the same AVX2 specialization for K=128, N=64 and previously K=N=1024.
The initial filtered sample report did not show the K=64, N=32 kernel; its exact symbol identity is not yet
established by that capture. Do not infer it solely from the matching count.
Follow-up evidence is at `/datadisks/disk10/tommasoadani/tmp/matmul-m1-followup-f90Hd9/`.

### Held-out M=1 validation

Three new geometries were generated with cohort `held_out`, preserving FP32 input, accuracy request 0,
INT4/block size 32, no bias, implicit midpoint zero points and one intra-op thread.
All three passed estimator capture/reporting checks. Three R/2R pairs per case, with R=1000 and 100 warm-up
runs per process, completed successfully (18 hardware measurements).

| K | N | Existing estimator FP subtotal | Source-derived kernel FP ops, 3NK + 7N | Measured process mean | Sample stddev | Residual |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 128 | 131,072 | 99,200 | 99,202.000 | 0.018 | +2.000 |
| 512 | 256 | 524,288 | 395,008 | 395,010.012 | 0.010 | +2.012 |
| 2,048 | 512 | 4,194,304 | 3,149,312 | 3,149,314.012 | 0.021 | +2.012 |

Cycle sampling with 100,000 workload repetitions confirmed
`SQ4BitGemmM1Kernel_BlkLen32Plus_CompFp32_avx2<false> [clone .isra.0]` for all three held-out cases.
Captures contained 4,461, 5,473 and 14,180 samples respectively. Kernel self-cycle attribution was 7.44%, 23.20%
and 70.98%; these percentages are not arithmetic counts and include the process-level sampling scope.

The approximately two-operation residual is negligible for a ballpark estimate and remains unattributed.
Fractional means/residuals arise from differential measurements and averaging, not fractional instructions.
The small nonzero standard deviations here describe the derived three-trial counts; they do not alter the
zero standard deviations reported for the original matrix.

This is strong held-out corroboration of one FP scaling operation per weight plus final reductions for the
qualified aligned M=1 AVX2 specialization. It does not qualify every `sqnbit_fp32` implementation or convert
process counters into isolated operator measurements. A future refinement should preserve core `2NK`, model
scaling `NK` only when the specialization is reliably identified, and account for reductions separately if
expanding scope. Do not tune a coefficient to eliminate the negligible residual.

Evidence is at `/datadisks/disk10/tommasoadani/tmp/matmul-m1-heldout-gw52tI/`:
`heldout-comparison.csv`, `estimates/results.json`, `hardware/metadata.json`, `hardware/summary.csv`,
raw trial counters and `samples/*/symbols.txt`. No production estimator formula has been changed.

## Follow-up audit: aligned INT8 AVX2 block-sum path (November 10)

The original audit case uses FP32 input `[32,128]`, N=64, accuracy request 4, INT4/block size 32,
no bias, implicit midpoint zero points and one intra-op thread. The installed native libraries matched
the saved baseline. Sampling identified:

- `QuantizeARow_CompInt8_avx2`: activation quantization and scaled activation block sums.
- `SQ4BitGemmKernel_BlkSum_CompInt8_avx2`: quantized computation and block-sum correction wrapper.
- `MlasGemmFloatKernelFma3`: floating-point correction GEMM.

The wrapper selects `MlasQ4Int8GemmKernelBlkLen32Avx2<false>` and its
`Q4Int8Gemm2x4x2BlkLen32Avx2<false>` helper for this even-row, four-column-aligned case. It processes
two K blocks at a time. Unlike the M=1 branch, the multi-row wrapper also calls a separate floating-point
GEMM over the activation/weight block sums.

Let G=K/32. The independent candidate ledger is:

| Component | Bounded candidate formula | Original case FP ops |
|---|---|---:|
| Activation quantization/scaled block sums | MK + 3MG | 4,480 |
| Vector scale combination and FP accumulation | 12MNG | 98,304 |
| Final quantized-kernel accumulator reductions | 7MN | 14,336 |
| Block-sum correction GEMM and output epilogue | 2MNG + 2MN | 20,480 |
| Candidate total | MK + 3MG + 14MNG + 9MN | **137,600** |
| Measured recurring process FP ops | | **137,602** |
| Unattributed residual | | **+2** |
| Existing estimator modeled subtotal | 5MNG | **40,960** |

The [two-block accumulation helper](../../../../onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen32.h)
performs an eight-lane multiplication and eight-lane FMA per output/two-block pair:
8 + 16 = 24 FP operations, or 12 per output/block. Integer dot products themselves contribute zero FP operations.
The [activation quantizer](../../../../onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2.cpp) performs activation
scaling and per-block scale/inverse-scale/scaled-sum arithmetic. The observed correction GEMM assembly includes
an FMA output epilogue that combines the correction with existing C, even with alpha=1.
Pure conversions and integer arithmetic are separate from the FP ledger.

This explains why activation quantization alone cannot account for the original discrepancy. The existing
five-operation nominal scaling/correction allowance is not a lane-level model. Do not add the candidate ledger
on top of that subtotal: scaling/correction work overlaps. The candidate quantizer count and hardware treatment
of rounding/max instructions still require compiled-instruction/event-semantics qualification. Matching a total
does not independently prove the allocation of operations to each phase.

Original audit evidence is at `/datadisks/disk10/tommasoadani/tmp/matmul-int8-audit-RsVHmE/`, including cycle samples,
symbol reports, source excerpts and kernel/quantizer/correction-GEMM annotations. The cycle capture had 4,735 samples.
Cycle percentages identify hot functions, not component FP counts.

### Held-out aligned INT8 validation

Three new `held_out` cases preserve FP32/accuracy-4, INT4/block size 32, no bias/explicit zero points and one thread.
Each has even M, N divisible by four and an even full K-block count, preserving the candidate branch's assumptions.
All passed estimator capture/reporting checks; three R/2R pairs per case with R=1000 and 100 warm-up runs
completed successfully (18 hardware measurements).

| M | K | N | G | Existing estimator subtotal | Candidate total | Measured process mean | Sample stddev | Residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 128 | 2 | 2,560 | 9,612 | 9,614.030 | 0.010 | +2.030 |
| 16 | 256 | 128 | 8 | 81,920 | 252,288 | 252,289.976 | 0.021 | +1.976 |
| 64 | 512 | 256 | 16 | 1,310,720 | 3,853,312 | 3,853,313.982 | 0.018 | +1.982 |

Sampling with 100,000 repetitions found the same AVX2 quantizer, block-sum wrapper and FMA3 correction kernel
in all three cases. Captures contained 4,236, 5,095 and 16,016 samples respectively. Matching wrapper names
corroborate family identity; internal branch applicability also depends on the validated geometry/source dispatch.

The approximately two-operation residual is negligible for a ballpark model and remains unattributed.
Fractional means reflect differential measurements/averaging, not fractional instructions.
These results strongly corroborate the bounded candidate total on held-out shapes, not a universal INT8 formula,
independently measured phase totals or isolated whole-operator accuracy.

Evidence is at `/datadisks/disk10/tommasoadani/tmp/matmul-int8-heldout-8EbLdy/`:
`heldout-comparison.csv`, estimator reports, `hardware/metadata.json`, trial counters/summaries and
`samples/*/symbols.txt`. Raw evidence remains outside the repository.

No estimator refinements were made, as requested. Next qualify M=1 INT8 separately: the wrapper takes a different
single-row branch and bypasses this separate correction GEMM. Odd block counts, row/column/K tails, other block
sizes/ISAs, bias and explicit-zero-point variants also require their own accounting.

## Results

| Mode | Input dtype | Accuracy request | M | K | N | Estimated FP ops | Measured mean | Sample stddev | Min | Max | Difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sqnbit_fp32 | float32 | 0 | 1 | 64 | 32 | 8,192 | 6,370.000 | 0.000 | 6,370.000 | 6,370.000 | -22.24% |
| sqnbit_fp32 | float32 | 0 | 1 | 128 | 64 | 32,768 | 25,026.000 | 0.000 | 25,026.000 | 25,026.000 | -23.63% |
| sqnbit_fp32 | float32 | 0 | 1 | 1024 | 1024 | 4,194,304 | 3,152,898.000 | 0.000 | 3,152,898.000 | 3,152,898.000 | -24.83% |
| sqnbit_fp32 | float32 | 0 | 8 | 64 | 32 | 36,864 | 35,074.000 | 0.000 | 35,074.000 | 35,074.000 | -4.86% |
| sqnbit_fp32 | float32 | 0 | 8 | 128 | 64 | 147,456 | 139,778.000 | 0.000 | 139,778.000 | 139,778.000 | -5.21% |
| sqnbit_fp32 | float32 | 0 | 8 | 1024 | 1024 | 18,874,368 | 17,833,986.000 | 0.000 | 17,833,986.000 | 17,833,986.000 | -5.51% |
| sqnbit_fp32 | float32 | 0 | 32 | 64 | 32 | 135,168 | 134,146.000 | 0.000 | 134,146.000 | 134,146.000 | -0.76% |
| sqnbit_fp32 | float32 | 0 | 32 | 128 | 64 | 540,672 | 534,530.000 | 0.000 | 534,530.000 | 534,530.000 | -1.14% |
| sqnbit_fp32 | float32 | 0 | 32 | 1024 | 1024 | 69,206,016 | 68,190,210.000 | 0.000 | 68,190,210.000 | 68,190,210.000 | -1.47% |
| sqnbit_fp32 | float32 | 0 | 128 | 64 | 32 | 528,384 | 530,434.000 | 0.000 | 530,434.000 | 530,434.000 | +0.39% |
| sqnbit_fp32 | float32 | 0 | 128 | 128 | 64 | 2,113,536 | 2,113,538.000 | 0.000 | 2,113,538.000 | 2,113,538.000 | +0.00% |
| sqnbit_fp32 | float32 | 0 | 128 | 1024 | 1024 | 270,532,608 | 269,615,106.000 | 0.000 | 269,615,106.000 | 269,615,106.000 | -0.34% |
| sqnbit_fp32 | float32 | 1 | 1 | 64 | 32 | 8,192 | 6,370.000 | 0.000 | 6,370.000 | 6,370.000 | -22.24% |
| sqnbit_fp32 | float32 | 1 | 1 | 128 | 64 | 32,768 | 25,026.000 | 0.000 | 25,026.000 | 25,026.000 | -23.63% |
| sqnbit_fp32 | float32 | 1 | 1 | 1024 | 1024 | 4,194,304 | 3,152,898.000 | 0.000 | 3,152,898.000 | 3,152,898.000 | -24.83% |
| sqnbit_fp32 | float32 | 1 | 8 | 64 | 32 | 36,864 | 35,074.000 | 0.000 | 35,074.000 | 35,074.000 | -4.86% |
| sqnbit_fp32 | float32 | 1 | 8 | 128 | 64 | 147,456 | 139,778.000 | 0.000 | 139,778.000 | 139,778.000 | -5.21% |
| sqnbit_fp32 | float32 | 1 | 8 | 1024 | 1024 | 18,874,368 | 17,833,986.000 | 0.000 | 17,833,986.000 | 17,833,986.000 | -5.51% |
| sqnbit_fp32 | float32 | 1 | 32 | 64 | 32 | 135,168 | 134,146.000 | 0.000 | 134,146.000 | 134,146.000 | -0.76% |
| sqnbit_fp32 | float32 | 1 | 32 | 128 | 64 | 540,672 | 534,530.000 | 0.000 | 534,530.000 | 534,530.000 | -1.14% |
| sqnbit_fp32 | float32 | 1 | 32 | 1024 | 1024 | 69,206,016 | 68,190,210.000 | 0.000 | 68,190,210.000 | 68,190,210.000 | -1.47% |
| sqnbit_fp32 | float32 | 1 | 128 | 64 | 32 | 528,384 | 530,434.000 | 0.000 | 530,434.000 | 530,434.000 | +0.39% |
| sqnbit_fp32 | float32 | 1 | 128 | 128 | 64 | 2,113,536 | 2,113,538.000 | 0.000 | 2,113,538.000 | 2,113,538.000 | +0.00% |
| sqnbit_fp32 | float32 | 1 | 128 | 1024 | 1024 | 270,532,608 | 269,615,106.000 | 0.000 | 269,615,106.000 | 269,615,106.000 | -0.34% |
| sqnbit_fp32 | float32 | 2 | 1 | 64 | 32 | 8,192 | 6,370.000 | 0.000 | 6,370.000 | 6,370.000 | -22.24% |
| sqnbit_fp32 | float32 | 2 | 1 | 128 | 64 | 32,768 | 25,026.000 | 0.000 | 25,026.000 | 25,026.000 | -23.63% |
| sqnbit_fp32 | float32 | 2 | 1 | 1024 | 1024 | 4,194,304 | 3,152,898.000 | 0.000 | 3,152,898.000 | 3,152,898.000 | -24.83% |
| sqnbit_fp32 | float32 | 2 | 8 | 64 | 32 | 36,864 | 35,074.000 | 0.000 | 35,074.000 | 35,074.000 | -4.86% |
| sqnbit_fp32 | float32 | 2 | 8 | 128 | 64 | 147,456 | 139,778.000 | 0.000 | 139,778.000 | 139,778.000 | -5.21% |
| sqnbit_fp32 | float32 | 2 | 8 | 1024 | 1024 | 18,874,368 | 17,833,986.000 | 0.000 | 17,833,986.000 | 17,833,986.000 | -5.51% |
| sqnbit_fp32 | float32 | 2 | 32 | 64 | 32 | 135,168 | 134,146.000 | 0.000 | 134,146.000 | 134,146.000 | -0.76% |
| sqnbit_fp32 | float32 | 2 | 32 | 128 | 64 | 540,672 | 534,530.000 | 0.000 | 534,530.000 | 534,530.000 | -1.14% |
| sqnbit_fp32 | float32 | 2 | 32 | 1024 | 1024 | 69,206,016 | 68,190,210.000 | 0.000 | 68,190,210.000 | 68,190,210.000 | -1.47% |
| sqnbit_fp32 | float32 | 2 | 128 | 64 | 32 | 528,384 | 530,434.000 | 0.000 | 530,434.000 | 530,434.000 | +0.39% |
| sqnbit_fp32 | float32 | 2 | 128 | 128 | 64 | 2,113,536 | 2,113,538.000 | 0.000 | 2,113,538.000 | 2,113,538.000 | +0.00% |
| sqnbit_fp32 | float32 | 2 | 128 | 1024 | 1024 | 270,532,608 | 269,615,106.000 | 0.000 | 269,615,106.000 | 269,615,106.000 | -0.34% |
| sqnbit_fp32 | float32 | 3 | 1 | 64 | 32 | 8,192 | 6,370.000 | 0.000 | 6,370.000 | 6,370.000 | -22.24% |
| sqnbit_fp32 | float32 | 3 | 1 | 128 | 64 | 32,768 | 25,026.000 | 0.000 | 25,026.000 | 25,026.000 | -23.63% |
| sqnbit_fp32 | float32 | 3 | 1 | 1024 | 1024 | 4,194,304 | 3,152,898.000 | 0.000 | 3,152,898.000 | 3,152,898.000 | -24.83% |
| sqnbit_fp32 | float32 | 3 | 8 | 64 | 32 | 36,864 | 35,074.000 | 0.000 | 35,074.000 | 35,074.000 | -4.86% |
| sqnbit_fp32 | float32 | 3 | 8 | 128 | 64 | 147,456 | 139,778.000 | 0.000 | 139,778.000 | 139,778.000 | -5.21% |
| sqnbit_fp32 | float32 | 3 | 8 | 1024 | 1024 | 18,874,368 | 17,833,986.000 | 0.000 | 17,833,986.000 | 17,833,986.000 | -5.51% |
| sqnbit_fp32 | float32 | 3 | 32 | 64 | 32 | 135,168 | 134,146.000 | 0.000 | 134,146.000 | 134,146.000 | -0.76% |
| sqnbit_fp32 | float32 | 3 | 32 | 128 | 64 | 540,672 | 534,530.000 | 0.000 | 534,530.000 | 534,530.000 | -1.14% |
| sqnbit_fp32 | float32 | 3 | 32 | 1024 | 1024 | 69,206,016 | 68,190,210.000 | 0.000 | 68,190,210.000 | 68,190,210.000 | -1.47% |
| sqnbit_fp32 | float32 | 3 | 128 | 64 | 32 | 528,384 | 530,434.000 | 0.000 | 530,434.000 | 530,434.000 | +0.39% |
| sqnbit_fp32 | float32 | 3 | 128 | 128 | 64 | 2,113,536 | 2,113,538.000 | 0.000 | 2,113,538.000 | 2,113,538.000 | +0.00% |
| sqnbit_fp32 | float32 | 3 | 128 | 1024 | 1024 | 270,532,608 | 269,615,106.000 | 0.000 | 269,615,106.000 | 269,615,106.000 | -0.34% |
| sqnbit_int8 | float32 | 4 | 1 | 64 | 32 | 320 | 1,064.000 | 0.000 | 1,064.000 | 1,064.000 | +232.50% |
| sqnbit_int8 | float32 | 4 | 1 | 128 | 64 | 1,280 | 3,662.000 | 0.000 | 3,662.000 | 3,662.000 | +186.09% |
| sqnbit_int8 | float32 | 4 | 1 | 1024 | 1024 | 163,840 | 401,506.000 | 0.000 | 401,506.000 | 401,506.000 | +145.06% |
| sqnbit_int8 | float32 | 4 | 8 | 64 | 32 | 2,560 | 10,034.000 | 0.000 | 10,034.000 | 10,034.000 | +291.95% |
| sqnbit_int8 | float32 | 4 | 8 | 128 | 64 | 10,240 | 34,402.000 | 0.000 | 34,402.000 | 34,402.000 | +235.96% |
| sqnbit_int8 | float32 | 4 | 8 | 1024 | 1024 | 1,310,720 | 3,752,706.000 | 0.000 | 3,752,706.000 | 3,752,706.000 | +186.31% |
| sqnbit_int8 | float32 | 4 | 32 | 64 | 32 | 10,240 | 40,130.000 | 0.000 | 40,130.000 | 40,130.000 | +291.89% |
| sqnbit_int8 | float32 | 4 | 32 | 128 | 64 | 40,960 | 137,602.000 | 0.000 | 137,602.000 | 137,602.000 | +235.94% |
| sqnbit_int8 | float32 | 4 | 32 | 1024 | 1024 | 5,242,880 | 15,010,818.000 | 0.000 | 15,010,818.000 | 15,010,818.000 | +186.31% |
| sqnbit_int8 | float32 | 4 | 128 | 64 | 32 | 40,960 | 160,514.000 | 0.000 | 160,514.000 | 160,514.000 | +291.88% |
| sqnbit_int8 | float32 | 4 | 128 | 128 | 64 | 163,840 | 550,402.000 | 0.000 | 550,402.000 | 550,402.000 | +235.94% |
| sqnbit_int8 | float32 | 4 | 128 | 1024 | 1024 | 20,971,520 | 60,043,266.000 | 0.000 | 60,043,266.000 | 60,043,266.000 | +186.31% |
| sqnbit_fp32 | float16 | 0 | 1 | 64 | 32 | 8,192 | 6,370.000 | 0.000 | 6,370.000 | 6,370.000 | -22.24% |
| sqnbit_fp32 | float16 | 0 | 1 | 128 | 64 | 32,768 | 25,026.000 | 0.000 | 25,026.000 | 25,026.000 | -23.63% |
| sqnbit_fp32 | float16 | 0 | 1 | 1024 | 1024 | 4,194,304 | 3,152,898.000 | 0.000 | 3,152,898.000 | 3,152,898.000 | -24.83% |
| sqnbit_fp32 | float16 | 0 | 8 | 64 | 32 | 36,864 | 35,074.000 | 0.000 | 35,074.000 | 35,074.000 | -4.86% |
| sqnbit_fp32 | float16 | 0 | 8 | 128 | 64 | 147,456 | 139,778.000 | 0.000 | 139,778.000 | 139,778.000 | -5.21% |
| sqnbit_fp32 | float16 | 0 | 8 | 1024 | 1024 | 18,874,368 | 17,833,986.000 | 0.000 | 17,833,986.000 | 17,833,986.000 | -5.51% |
| sqnbit_fp32 | float16 | 0 | 32 | 64 | 32 | 135,168 | 134,146.000 | 0.000 | 134,146.000 | 134,146.000 | -0.76% |
| sqnbit_fp32 | float16 | 0 | 32 | 128 | 64 | 540,672 | 534,530.000 | 0.000 | 534,530.000 | 534,530.000 | -1.14% |
| sqnbit_fp32 | float16 | 0 | 32 | 1024 | 1024 | 69,206,016 | 68,190,210.000 | 0.000 | 68,190,210.000 | 68,190,210.000 | -1.47% |
| sqnbit_fp32 | float16 | 0 | 128 | 64 | 32 | 528,384 | 530,434.000 | 0.000 | 530,434.000 | 530,434.000 | +0.39% |
| sqnbit_fp32 | float16 | 0 | 128 | 128 | 64 | 2,113,536 | 2,113,538.000 | 0.000 | 2,113,538.000 | 2,113,538.000 | +0.00% |
| sqnbit_fp32 | float16 | 0 | 128 | 1024 | 1024 | 270,532,608 | 269,615,106.000 | 0.000 | 269,615,106.000 | 269,615,106.000 | -0.34% |
| sqnbit_int8 | float16 | 4 | 1 | 64 | 32 | 320 | 1,064.000 | 0.000 | 1,064.000 | 1,064.000 | +232.50% |
| sqnbit_int8 | float16 | 4 | 1 | 128 | 64 | 1,280 | 3,662.000 | 0.000 | 3,662.000 | 3,662.000 | +186.09% |
| sqnbit_int8 | float16 | 4 | 1 | 1024 | 1024 | 163,840 | 401,506.000 | 0.000 | 401,506.000 | 401,506.000 | +145.06% |
| sqnbit_int8 | float16 | 4 | 8 | 64 | 32 | 2,560 | 10,034.000 | 0.000 | 10,034.000 | 10,034.000 | +291.95% |
| sqnbit_int8 | float16 | 4 | 8 | 128 | 64 | 10,240 | 34,402.000 | 0.000 | 34,402.000 | 34,402.000 | +235.96% |
| sqnbit_int8 | float16 | 4 | 8 | 1024 | 1024 | 1,310,720 | 3,752,706.000 | 0.000 | 3,752,706.000 | 3,752,706.000 | +186.31% |
| sqnbit_int8 | float16 | 4 | 32 | 64 | 32 | 10,240 | 40,130.000 | 0.000 | 40,130.000 | 40,130.000 | +291.89% |
| sqnbit_int8 | float16 | 4 | 32 | 128 | 64 | 40,960 | 137,602.000 | 0.000 | 137,602.000 | 137,602.000 | +235.94% |
| sqnbit_int8 | float16 | 4 | 32 | 1024 | 1024 | 5,242,880 | 15,010,818.000 | 0.000 | 15,010,818.000 | 15,010,818.000 | +186.31% |
| sqnbit_int8 | float16 | 4 | 128 | 64 | 32 | 40,960 | 160,514.000 | 0.000 | 160,514.000 | 160,514.000 | +291.88% |
| sqnbit_int8 | float16 | 4 | 128 | 128 | 64 | 163,840 | 550,402.000 | 0.000 | 550,402.000 | 550,402.000 | +235.94% |
| sqnbit_int8 | float16 | 4 | 128 | 1024 | 1024 | 20,971,520 | 60,043,266.000 | 0.000 | 60,043,266.000 | 60,043,266.000 | +186.31% |
