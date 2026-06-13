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
