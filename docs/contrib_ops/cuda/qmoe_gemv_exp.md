# QMoE GEMV Profiling Experiments

This file records QMoE INT4 per-channel GEMV profiling results so future kernel and dispatch changes can be compared against a stable baseline.

## 2026-06-12 Baseline: SM90, Warmup 5, Repeat 100

### Setup

- Machine/GPU: local CUDA machine, SM90 reported by `torch.cuda.get_device_capability()`.
- ONNX Runtime build: `/home/tianlei/onnxruntime/build/cu130/Release`.
- Python: `/home/tianlei/onnxruntime/.venv_cu130/bin/python`.
- Nsight Systems: `/home/tianlei/cuda13.0/bin/nsys`.
- Benchmark script: `onnxruntime/test/python/transformers/profile_qmoe_gemv.sh`.
- Warmup: 5 ORT runs before the measured `benchmark` NVTX range.
- Repeat: 100 measured ORT runs inside the `benchmark` NVTX range.
- Common shape settings: hidden size 128, intermediate size 256, 4 experts, top-k 2, INT4 symmetric per-channel weights, SwiGLU interleaved activation.
- Profile log: `/tmp/qmoe_gemv_warmup5_r100.log`.
- Nsight artifacts: `/tmp/qmoe_gemv_<case>_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.

Command template:

```bash
pushd /tmp >/dev/null
PATH=/home/tianlei/cuda13.0/bin:$PATH \
PYTHONPATH=/home/tianlei/onnxruntime/build/cu130/Release:/home/tianlei/onnxruntime/onnxruntime/test/python/transformers \
/home/tianlei/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
  --python /home/tianlei/onnxruntime/.venv_cu130/bin/python \
  --case <case-name> --warmup 5 --repeat 100 \
  -o /tmp/qmoe_gemv_<case-name>_warmup5_r100
popd >/dev/null
```

Parse command used for kernel-level comparisons:

```bash
python onnxruntime/test/python/transformers/parse_nsys.py \
  /tmp/qmoe_gemv_<case>_warmup5_r100_<mode>.sqlite \
  --nvtx-range benchmark --pattern '%'
```

`--pattern '%'` is important for fallback runs because the grouped-GEMM compute kernels are named as CUTLASS kernels and are not shown by the default parser pattern.

### End-To-End ORT Loop Timing

Lower is better. `GEMV/GEMM` compares the host-observed average latency printed by `profile_qmoe_gemv.py`.

| Case | Tokens | Expanded rows | DType | GEMV ms | GEMM fallback ms | GEMV/GEMM | Result |
|------|--------|---------------|-------|---------|------------------|-----------|--------|
| `m1_top2_fp16_128x256` | 1 | 2 | FP16 | 0.0967 | 0.0706 | 1.37x | GEMV slower |
| `m4_top2_fp16_128x256` | 4 | 8 | FP16 | 0.1372 | 0.0705 | 1.94x | GEMV slower |
| `m8_top2_fp16_128x256` | 8 | 16 | FP16 | 0.1381 | 0.0702 | 1.97x | GEMV slower |
| `m1_top2_bf16_128x256` | 1 | 2 | BF16 | 0.0972 | 0.0706 | 1.38x | GEMV slower |

### Primary Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX range. GEMV has two compute kernels per ORT run in these cases, corresponding to the two MoE FC stages. GEMM fallback has two grouped-GEMM kernels per ORT run, except `m8_top2_fp16_128x256` appears as one aggregated CUTLASS kernel row with 200 calls.

| Case | GEMV compute avg us | GEMM compute avg us | Notes |
|------|---------------------|---------------------|-------|
| `m1_top2_fp16_128x256` | 2.82, 2.56 | 3.98, 3.44 | GEMV compute kernels are faster, but total ORT loop is slower. |
| `m4_top2_fp16_128x256` | 3.01, 2.74 | 3.72, 3.42 | GEMV compute kernels are faster, but total ORT loop is slower. |
| `m8_top2_fp16_128x256` | 3.09, 2.75 | 3.55 aggregated over 200 calls | GEMV compute kernels are faster per call, but total ORT loop is slower. |
| `m1_top2_bf16_128x256` | 2.84, 2.61 | 4.22, 3.49 | GEMV compute kernels are faster, but total ORT loop is slower. |

### Observations

- With explicit warmup and `nsys` NVTX filtering, the host-observed ORT loop latency is not aligned with the earlier low-repeat smoke numbers that suggested a GEMV win. This warmed baseline shows grouped GEMM fallback faster for these small 128x256 cases.
- Kernel-only compute timing still shows the custom GEMV compute kernels faster than grouped GEMM compute kernels. The end-to-end loss likely comes from non-compute overhead around the GEMV path, dispatch/prologue behavior, extra launches, or measurement sensitivity at very small latencies.
- The default `parse_nsys.py` kernel filter can hide fallback CUTLASS grouped-GEMM kernels. Use `--pattern '%'` for GEMV-vs-GEMM profile comparisons.
- This baseline strengthens the P1/P2 work items: avoid repeated row-to-expert scans, tune tile/threshold decisions with data, and consider fusing FC1 activation or FC2 finalize/scatter before expanding GEMV coverage.

### Next Experiments

- Sweep larger realistic FC dimensions, especially model-like hidden/intermediate sizes, because this baseline only covers 128x256 test-sized shapes.
- Sweep expanded rows `{1, 2, 4, 8, 16, 32, 64}` while keeping shape and dtype fixed.
- Capture CUDA API timing inside the `benchmark` range to identify host-side synchronization or launch overhead differences.
- Record per-architecture results for SM80/SM89/SM90/SM100/SM120 when available.

## 2026-06-12 Larger Shape Sweep: SM90, 1024x4096, Warmup 5, Repeat 100

### Setup

- Same environment as the baseline section.
- Custom profiler arguments were used instead of named cases.
- Common shape settings: hidden size 1024, intermediate size 4096, 8 experts, top-k 2, INT4 symmetric per-channel weights, SwiGLU interleaved activation.
- Profile log: `/tmp/qmoe_gemv_1024x4096_e8_warmup5_r100.log`.
- Nsight artifacts: `/tmp/qmoe_gemv_custom_<case>_1024x4096_e8_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.

Command template:

```bash
pushd /tmp >/dev/null
PATH=/home/tianlei/cuda13.0/bin:$PATH \
PYTHONPATH=/home/tianlei/onnxruntime/build/cu130/Release:/home/tianlei/onnxruntime/onnxruntime/test/python/transformers \
/home/tianlei/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
  --python /home/tianlei/onnxruntime/.venv_cu130/bin/python \
  --batch-size 1 --sequence-length <m> \
  --hidden-size 1024 --intermediate-size 4096 \
  --num-experts 8 --top-k 2 --dtype FLOAT16 \
  --warmup 5 --repeat 100 \
  -o /tmp/qmoe_gemv_custom_m<m>_top2_float16_1024x4096_e8_warmup5_r100
popd >/dev/null
```

### End-To-End ORT Loop Timing

Lower is better. `GEMV/GEMM` compares the host-observed average latency printed by `profile_qmoe_gemv.py`.

| Case | Tokens | Expanded rows | DType | GEMV ms | GEMM fallback ms | GEMV/GEMM | Result |
|------|--------|---------------|-------|---------|------------------|-----------|--------|
| `custom_m1_top2_float16_1024x4096_e8` | 1 | 2 | FP16 | 0.0622 | 0.0811 | 0.77x | GEMV faster |
| `custom_m4_top2_float16_1024x4096_e8` | 4 | 8 | FP16 | 0.1542 | 0.0873 | 1.77x | GEMV slower |
| `custom_m8_top2_float16_1024x4096_e8` | 8 | 16 | FP16 | 0.2083 | 0.0976 | 2.13x | GEMV slower |
| `custom_m1_top2_bfloat16_1024x4096_e8` | 1 | 2 | BF16 | 0.2144 | 0.0844 | 2.54x | GEMV slower |

### Primary Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX range.

| Case | GEMV compute avg us | GEMM compute avg us | Notes |
|------|---------------------|---------------------|-------|
| `custom_m1_top2_float16_1024x4096_e8` | 6.78, 4.70 | 14.61, 8.55 | GEMV wins both compute and end-to-end. |
| `custom_m4_top2_float16_1024x4096_e8` | 12.05, 10.00 | 16.56, 12.71 | GEMV compute is faster, but total ORT loop is slower. |
| `custom_m8_top2_float16_1024x4096_e8` | 23.19, 13.50 | 26.34, 14.29 | GEMV compute advantage narrows and end-to-end latency is worse. |
| `custom_m1_top2_bfloat16_1024x4096_e8` | 6.99, 4.69 | 12.86 aggregated over 200 calls | BF16 GEMV is much slower end-to-end despite comparable compute-kernel timing. |

### Observations

- FP16 GEMV shows a real end-to-end win for single-token decode at 1024x4096, unlike the tiny 128x256 test shape.
- The previous broad `expanded_num_rows <= 64` dispatch threshold was too optimistic on SM90 for this shape. At expanded rows 8 and 16, grouped GEMM fallback is faster end-to-end.
- BF16 needs separate treatment. The single-token BF16 point is much slower with GEMV even though the raw GEMV kernels are not obviously bad.
- The initial P1 dispatch cutoff is therefore conservative: FP16 only, `expanded_num_rows <= 2`, and `N/K >= 1024`. Collect more model-size points around expanded rows 2, 4, 8, and 16 before enabling GEMV beyond true single-token decode.

## 2026-06-12 Post-Threshold Dispatch Smoke: SM90, Warmup 1, Repeat 2

### Setup

- Same build and Python environment as above, after syncing the rebuilt CUDA provider into the Python package `capi` directory.
- Nsight Systems profiles used the measured `benchmark` NVTX range and counted kernels matching `%moe_gemv%`.
- These runs validate routing only; repeat count is too small for performance comparison.

| Case | Expanded rows | DType | Expected route | `moe_gemv` calls in benchmark | Result |
|------|---------------|-------|----------------|--------------------------------|--------|
| `m1_top2_fp16_128x256` | 2 | FP16 | grouped GEMM fallback (`N/K < 1024`) | 0 | Passed |
| `custom_m1_top2_float16_1024x4096_e8` | 2 | FP16 | GEMV | 4 | Passed |
| `custom_m4_top2_float16_1024x4096_e8` | 8 | FP16 | grouped GEMM fallback (`expanded_num_rows > 2`) | 0 | Passed |
| `custom_m1_top2_bfloat16_1024x4096_e8` | 2 | BF16 | grouped GEMM fallback (FP16-only GEMV gate) | 0 | Passed |

## 2026-06-12 Actual Model Dimensions: SM90, FP16, Warmup 5, Repeat 20

### Setup

- Same build and Python environment as above.
- Nsight Systems profiles used the measured `benchmark` NVTX range.
- Enabled mode leaves `ORT_DISABLE_MOE_GEMV` unset. Fallback mode sets `ORT_DISABLE_MOE_GEMV=1`.
- Nsight artifacts: `/tmp/qmoe_actual_<case>_warmup5_r20_{enabled,gemm}.{nsys-rep,sqlite}`.
- Summary log: `/tmp/qmoe_actual_model_dims_fp16_warmup5_r20_summary.txt`.

Model dimensions came from the Hugging Face configs below. Qwen's shared expert
is ignored in this benchmark.

| Model case | Source dimensions | Tokens | Expanded rows | Enabled ms | Fallback ms | `moe_gemv` calls in enabled profile | Route |
|------------|-------------------|--------|---------------|------------|-------------|--------------------------------------|-------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | `hidden_size=2880`, `intermediate_size=2880`, `num_local_experts=32`, `top_k=4` | 1 | 4 | 0.0909 | 0.0944 | 0 | grouped GEMM fallback |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | `hidden_size=2048`, `moe_intermediate_size=512`, `num_experts=256`, `top_k=8` | 1 | 8 | 0.0744 | 0.0743 | 0 | grouped GEMM fallback |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | `hidden_size=2816`, `moe_intermediate_size=704`, `num_experts=128`, `top_k=8` | 1 | 8 | 0.0819 | 0.0820 | 0 | grouped GEMM fallback |

### Observations

- The actual single-token top-k values for these models produce expanded rows 4
  or 8, so they intentionally stay on grouped GEMM under the current
  `expanded_num_rows <= 2` GEMV gate.
- Enabled and forced-fallback timings are nearly identical, and the enabled
  profiles contain zero custom `moe_gemv` kernels, confirming the dispatch route.
- These profiles establish the current grouped-GEMM baseline for actual model
  dimensions. Future GEMV work for model-realistic top-k needs targeted row-count
  improvements before expanding the dispatch gate.

## 2026-06-12 Actual Model Dimensions With Relaxed GEMV Gate: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Dispatch was temporarily relaxed to `expanded_num_rows <= 8` and `N/K >= 512`
  to force GEMV coverage for GPT-OSS-20B, Qwen3.6-35B-A3B, and Gemma-4-26B-A4B
  FP16 model-size cases.
- Nsight artifacts: `/tmp/qmoe_actual_<case>_relaxed_warmup5_r100_{enabled,gemm}.{nsys-rep,sqlite}`.
- Summary log: `/tmp/qmoe_actual_model_dims_fp16_relaxed_warmup5_r100_summary.txt`.

| Model case | Expanded rows | Enabled route | Enabled ms | Fallback ms | GEMV/GEMM | GEMV calls | Primary enabled compute avg us | Primary fallback compute avg us | Decision |
|------------|---------------|---------------|------------|-------------|-----------|------------|--------------------------------|---------------------------------|----------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 4 | GEMV | 0.0760 | 0.0905 | 0.84x | 200 | 15.35, 11.94 | 20.11, 18.48 | Keep GEMV candidate |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | 8 | GEMV | 0.0865 | 0.0707 | 1.22x | 200 | 20.37, 18.77 | 10.33 aggregated over 200 calls | Reject current GEMV gate |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 8 | GEMV | 0.0745 | 0.0801 | 0.93x | 200 | 14.61, 11.85 | 13.58 aggregated over 200 calls | Keep GEMV candidate |

### Observations

- GPT-OSS-20B benefits from GEMV at expanded rows 4 with square 2880x2880 FC
  dimensions.
- Gemma-4-26B-A4B shows a smaller but positive end-to-end win at expanded rows 8
  with 2816x704 / 704x2816 FC dimensions.
- Qwen3.6-35B-A3B regresses with the current GEMV kernel at expanded rows 8 and
  2048x512 / 512x2048 FC dimensions. The primary GEMV kernels are slower than
  grouped GEMM for this 512-wide MoE hidden size.
- Based on this pass, the dispatch gate should keep `expanded_num_rows <= 8` and
  `N/K >= 512`, but require the logical MoE intermediate size and each GEMV call
  dimension to be at least 704 when `expanded_num_rows > 4`. The logical
  intermediate-size check is needed because Qwen's gated FC1 has `N=1024` even
  though the MoE hidden size is 512. This keeps GPT-OSS and Gemma enabled while
  routing the measured Qwen regression to grouped GEMM. A future autotuner could
  replace this hand cutoff.

### Final Route Check After Logical Intermediate-Size Guard

After adding the logical intermediate-size guard, short Nsight profiles with
warmup 1 and repeat 2 confirmed the intended final routing:

| Model case | Expected route | `moe_gemv` calls in benchmark | Result |
|------------|----------------|--------------------------------|--------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | GEMV | 4 | Passed |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | grouped GEMM fallback | 0 | Passed |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | GEMV | 4 | Passed |

## 2026-06-13 P1 Row-To-Expert Map: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Change under test: the prologue now writes the local expert id for each
  permuted row into `permuted_token_selected_experts`, and the INT4
  per-channel MoE GEMV kernel uses that direct row-to-expert map instead of
  scanning `expert_first_token_offset` in every N-tile CTA. The prefix-offset
  scan remains as a fallback when no map is passed.
- Nsight artifacts:
  `/tmp/qmoe_actual_gpt_oss_20b_p1_row_expert_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`
  and
  `/tmp/qmoe_actual_gemma4_26b_p1_row_expert_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.
- Summary logs:
  `/tmp/qmoe_actual_gpt_oss_20b_p1_row_expert_warmup5_r100.log` and
  `/tmp/qmoe_actual_gemma4_26b_p1_row_expert_warmup5_r100.log`.

### End-To-End ORT Loop Timing

| Model case | Expanded rows | Enabled route | Enabled ms | Fallback ms | GEMV/GEMM | GEMV calls | Result |
|------------|---------------|---------------|------------|-------------|-----------|------------|--------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 4 | GEMV | 0.0721 | 0.0941 | 0.77x | 200 | GEMV faster |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 8 | GEMV | 0.0610 | 0.0795 | 0.77x | 200 | GEMV faster |

### Primary Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX
range. The two GEMV compute rows correspond to FC1 and FC2.

| Model case | GEMV compute avg us | Fallback compute avg us | Previous GEMV compute avg us | Notes |
|------------|---------------------|-------------------------|------------------------------|-------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 13.69, 10.22 | 20.32, 18.55 | 15.35, 11.94 | Direct map reduces GEMV kernel time by about 11% and 14% versus the relaxed-gate baseline. |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 7.17, 4.56 | 18.82, 7.99 | 14.61, 11.85 | Direct map removes a large per-tile scan cost at top-k 8 for this 704-wide case. |

### Observations

- The P1 row-to-expert map improves both actual-model GEMV candidates without
  changing the dispatch policy.
- The improvement is largest for Gemma, where expanded rows 8 and many N tiles
  made the repeated prefix scan particularly visible.
- Qwen remains routed to grouped GEMM by the logical-intermediate-size guard. A
  quick enabled-mode smoke run reported valid output for
  `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256`.

## 2026-06-12 P1 Tile-Shape Probe: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Goal: test the next P1 tile-shape knobs against the actual model-size cases
  that route to GEMV after the row-to-expert map optimization.
- Variants tested:
  - `CtaN=16, Threads=128`: halves the number of N-tile CTAs.
  - `CtaN=8, Threads=64`: reduces threads per CTA while preserving the N tile.
  - `CtaN=8, Threads=128` with a map-specialized launch: splits direct-map and
    prefix-scan kernels so the common direct-map path avoids the runtime branch.
- Nsight artifacts:
  `/tmp/qmoe_actual_*_ctan16_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`,
  `/tmp/qmoe_actual_*_threads64_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`,
  and `/tmp/qmoe_actual_*_specialized_map_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.

### End-To-End ORT Loop Timing

| Variant | Model case | Enabled ms | Fallback ms | Result |
|---------|------------|------------|-------------|--------|
| `CtaN=16, Threads=128` | `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 0.0852 | 0.0872 | Reject |
| `CtaN=16, Threads=128` | `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 0.0644 | 0.0791 | Reject |
| `CtaN=8, Threads=64` | `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 0.0832 | 0.0993 | Reject |
| `CtaN=8, Threads=64` | `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 0.0645 | 0.0799 | Reject |
| map-specialized launch | `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 0.0726 | 0.0909 | Reject, neutral/slightly worse kernels |
| map-specialized launch | `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 0.0600 | 0.0790 | Reject, neutral/slightly worse kernels |

### Primary GEMV Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX
range. Previous best is the row-to-expert map build with `CtaN=8, Threads=128`:
GPT `13.69, 10.22` us and Gemma `7.17, 4.56` us.

| Variant | GPT GEMV avg us | Gemma GEMV avg us | Decision |
|---------|-----------------|-------------------|----------|
| `CtaN=16, Threads=128` | 19.48, 17.52 | 11.07, 5.65 | Wider N tile is slower for both models. |
| `CtaN=8, Threads=64` | 18.77, 16.78 | 11.09, 5.09 | Fewer threads are slower for both models. |
| map-specialized launch | 14.00, 10.10 | 7.25, 4.62 | No clear kernel win; keep the simpler launch. |

### Observations

- The existing `CtaN=8, Threads=128` tile remains the best measured choice for
  the current SM90 actual-model cases.
- Halving N tiles with `CtaN=16` reduced CTA count but lost too much per-CTA
  efficiency.
- `Threads=64` did not help the 704-wide Gemma case and significantly hurt the
  larger GPT-OSS case.
- The direct-map branch in the current kernel is not a visible bottleneck after
  the row-to-expert map optimization, so the extra specialized launch variants
  are not worth the additional code size.
