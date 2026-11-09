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
