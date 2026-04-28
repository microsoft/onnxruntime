# RISC-V MLAS Benchmarks

This directory stores the standalone benchmarks and compare tools used while
bringing up and tuning the RVV path in MLAS.

Files:

- `sgemm_riscv_bench.cpp`: standalone SGEMM timing harness with checksum
  output. Useful for RVV versus scalar comparisons.
- `softmax_rvv_compare.cpp`: scalar versus RVV validation and timing tool for
  the Softmax critical path.

These tools are intentionally kept separate from `onnxruntime_mlas_benchmark`.
Each source file has its own `main()` and is built as an independent target.

## Build

On a riscv64 RVV build, first regenerate the build tree:

```bash
python3 tools/ci_build/build.py \
  --config Release \
  --build_dir build/k1_rvv_resync \
  --update \
  --skip_tests \
  --skip_pip_install \
  --skip_submodule_sync \
  --no_sve \
  --enable_rvv
```

Then build both standalone tools directly with CMake:

```bash
cmake --build build/k1_rvv_resync/Release \
  --config Release \
  --target onnxruntime_mlas_sgemm_riscv_bench onnxruntime_mlas_softmax_riscv_compare \
  -- -j8
```

The resulting binaries are typically placed under:

```bash
build/k1_rvv_resync/Release/onnxruntime_mlas_sgemm_riscv_bench
build/k1_rvv_resync/Release/onnxruntime_mlas_softmax_riscv_compare
```

## SGEMM examples

RVV:

```bash
taskset -c 0 build/k1_rvv_resync/Release/onnxruntime_mlas_sgemm_riscv_bench \
  --m=128 --n=3072 --k=768 --iters=10 --warmup=3 --trans_a=0 --trans_b=0
```

Scalar baseline on the same binary:

```bash
ORT_MLAS_RISCV_FORCE_SCALAR=1 taskset -c 0 \
  build/k1_rvv_resync/Release/onnxruntime_mlas_sgemm_riscv_bench \
  --m=128 --n=3072 --k=768 --iters=10 --warmup=3 --trans_a=0 --trans_b=0
```

## Softmax examples

```bash
taskset -c 0 build/k1_rvv_resync/Release/onnxruntime_mlas_softmax_riscv_compare
```

## Notes

- This reduced RVV benchmark set focuses on the SGEMM compute kernel and the
  Softmax critical path. It does not include a dedicated RVV pack-B helper or
  pack-B benchmark coverage.
- The RVV SGEMM and Softmax paths are written to be VLEN-agnostic and use
  runtime `vsetvl` chunking so the same binary works across different VLENs
  such as 128 and 256.
- `ORT_MLAS_RISCV_FORCE_SCALAR=1` disables the RVV dispatch at runtime and is
  the preferred way to gather scalar baselines from the same build.
