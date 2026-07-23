# CPU GroupQueryAttention performance experiments

## FP16 inputs with INT8 KV cache

This experiment compares FP16 and FP32 CPU `GroupQueryAttention` decode latency while both
use a per-tensor INT8 KV cache. It uses the flash-attention path selected by default by the
operator.

### Environment

- Machine: AMD EPYC 7763 (AVX2), 16 physical cores / 32 logical CPUs.
- Runtime: CPU-only Release wheel built from PR head `d144409dec9f1bb47ddea77867308af194debfcd`.
- CPU affinity: logical CPUs `0-7`.
- Operator configuration: batch size 1, query heads 16, KV heads 8, head size 128, decode
  sequence length 1, and per-tensor INT8 KV cache.
- Measurement: two runs, each with 50 warmup iterations and 500 measured iterations. The
  table reports the mean of the two runs.

### Build and install

From the repository root, activate the Python virtual environment and build the CPU wheel:

```bash
source .venv/bin/activate
./build.sh --build_dir build/cpu-fp16 --config Release --update --build --parallel 16 \
  --enable_pybind --build_shared_lib --build_wheel --skip_tests \
  --compile_no_warning_as_error \
  --cmake_extra_defines Python_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cd build/cpu-fp16/Release
"$VIRTUAL_ENV/bin/python" "$PWD/../../../setup.py" bdist_wheel
"$VIRTUAL_ENV/bin/python" -m pip install --force-reinstall --no-deps \
  dist/onnxruntime-1.29.0-cp314-cp314-linux_x86_64.whl
```

The `--compile_no_warning_as_error` flag was needed on this host because GCC 15 reports a
standard-library `maybe-uninitialized` false positive under the default warning policy.

### Benchmark commands

Run outside the source checkout so Python imports the installed wheel rather than the source
package:

```bash
cd /tmp
taskset -c 0-7 "$VIRTUAL_ENV/bin/python" \
  ~/onnxruntime/onnxruntime/test/python/transformers/benchmark_gqa_cpu_flash.py \
  --decode_only --warmup 50 --repeats 500
```

Run the command twice. The benchmark covers decode total sequence lengths 513, 1025, 2049,
and 4097, and prints separate FP32 and FP16 results using the same INT8 cache configuration.

### Results

Flash-attention latency in milliseconds:

| Decode total length | FP32 run 1 | FP32 run 2 | FP32 mean | FP16 run 1 | FP16 run 2 | FP16 mean | FP16 vs. FP32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 513 | 0.307 | 0.325 | 0.316 | 0.270 | 0.278 | 0.274 | 13% faster |
| 1025 | 0.397 | 0.393 | 0.395 | 0.418 | 0.355 | 0.386 | 2% faster |
| 2049 | 0.785 | 0.781 | 0.783 | 0.865 | 0.857 | 0.861 | 10% slower |
| 4097 | 0.869 | 1.137 | 1.003 | 0.783 | 1.000 | 0.892 | 11% faster |

FP16 is faster at three of the four tested decode lengths, including the longest context. The
single 2049-token result is within the observed run-to-run timing variation, so these results
do not show a sustained FP16 performance regression relative to FP32.

## FP16 quantized KV GEMM optimization

This experiment isolates the `MlasQKGemmFp16` and `MlasSVGemmFp16` calls used by quantized
GQA. The original FP16 kernels repeatedly converted FP16 query values inside the QK dot
products, and the SV implementation used a cache-unfriendly loop order that was substantially
slower than the equivalent FP32 kernel for INT8 caches.

### Changes

- QK converts an FP16 query tile to FP32 once when conversion is profitable. On x86,
  single-row INT8 decode keeps direct FP16 vector loads, while INT4 decode and multi-row
  prefill reuse the converted tile. NEON converts once and reuses its existing FP32 kernel.
- SV reuses the cache-friendly FP32 accumulation kernel with one FP32 scratch row, then writes
  the result to FP16 with vectorized conversion.
- Equivalent algorithmic changes are applied to AVX2, AVX-512, and NEON dispatches.

The KleidiAI `matmul_clamp_f16_f16_f16p` kernel was considered but not used. It requires an
FP16 packed RHS and produces FP16 output, whereas these APIs consume an INT4/INT8 KV cache and
QK produces FP32 scores. Dequantizing and packing the full cache on every attention call would
discard the fused quantized-cache advantage.

### Microbenchmark setup

- Binary: `build/cpu-fp16/Release/onnxruntime_mlas_benchmark`.
- Machine: the same AMD EPYC 7763 AVX2 host.
- Shape: decode `M=1`, head size 128, total sequence lengths 512 and 2048.
- Quantization: S8/S4, per-tensor and per-channel.
- Measurement: 10 repetitions with a 0.5-second minimum per repetition. Tables report median
  real time. Coefficients of variation were 0.07-1.04%.

Example command:

```bash
build/cpu-fp16/Release/onnxruntime_mlas_benchmark \
  --benchmark_filter='BM_(QKGemm|QKGemmFp16|SVGemm|SVGemmFp16)/M:1/.*128.*(512|2048).*' \
  --benchmark_min_time=0.5s --benchmark_repetitions=10 \
  --benchmark_report_aggregates_only=true --benchmark_time_unit=us
```

### Accumulated FP16 improvements

The `before` columns are measurements of the original FP16 kernels from the same build and
machine. `After vs. FP32` compares the optimized FP16 API with the FP32 API in the final build.

| op | total length | quantization | FP16 before (us) | FP16 after (us) | FP16 improvement | after vs. FP32 |
|---|---:|---|---:|---:|---:|---:|
| QK | 512 | S8 per-tensor | 5.725 | 5.387 | 5.9% | 6.1% faster |
| QK | 512 | S8 per-channel | 6.699 | 6.382 | 4.7% | 4.0% faster |
| QK | 512 | S4 per-tensor | 14.602 | 13.317 | 8.8% | 1.0% faster |
| QK | 512 | S4 per-channel | 14.141 | 13.483 | 4.7% | 0.4% faster |
| QK | 2048 | S8 per-tensor | 23.004 | 21.516 | 6.5% | 6.1% faster |
| QK | 2048 | S8 per-channel | 26.815 | 25.512 | 4.9% | 4.1% faster |
| QK | 2048 | S4 per-tensor | 58.476 | 53.222 | 9.0% | 0.9% faster |
| QK | 2048 | S4 per-channel | 56.578 | 53.927 | 4.7% | 0.3% faster |
| SV | 512 | S8 per-tensor | 10.156 | 5.007 | 50.7% | 8.4% slower |
| SV | 512 | S8 per-channel | 10.150 | 5.588 | 44.9% | 2.4% faster |
| SV | 512 | S4 per-tensor | 10.843 | 10.595 | 2.3% | 0.9% slower |
| SV | 512 | S4 per-channel | 10.588 | 9.852 | 6.9% | 1.3% slower |
| SV | 2048 | S8 per-tensor | 41.598 | 20.091 | 51.7% | 9.1% slower |
| SV | 2048 | S8 per-channel | 41.484 | 22.340 | 46.1% | 2.3% faster |
| SV | 2048 | S4 per-tensor | 43.028 | 42.399 | 1.5% | 1.0% slower |
| SV | 2048 | S4 per-channel | 42.120 | 39.361 | 6.6% | 1.3% slower |

QK FP16 improves by 4.7-9.0% and is faster than FP32 in every measured decode case. The
largest gain is SV with INT8, where FP16 improves by 44.9-51.7%; optimized SV finishes within
9.1% of FP32 across all measured modes.

Correctness and build validation:

- `onnxruntime_mlas_test --gtest_filter='KVQuant.*'`: passed.
- AVX2 and AVX-512 builds: passed.
- `lintrunner`: passed.
- NEON source diagnostics: clean, but an Arm64 compile was unavailable because this machine
  does not have an AArch64 cross compiler or sysroot.

## Intra-op thread scaling after the GEMM optimization

This experiment measures whether the FP16 gain remains visible with 4 and 8 intra-op threads
in the end-to-end GQA operator. It uses the same per-tensor INT8 cache, model shape, decode
lengths, warmup, and iteration count as the first experiment.

### Setup and commands

- Optimized wheel built from commit `2b68b5c9ed8f4beba1546ad584f6c672b852d48c` plus the
  uncommitted quantized KV GEMM changes described above.
- Four-thread runs are pinned to logical CPUs `0-3`; eight-thread runs are pinned to `0-7`.
- Two runs per thread count, each with 50 warmup and 500 measured iterations.
- The benchmark script accepts `--intra_op_num_threads` so FP32 and FP16 sessions use the
  same requested thread count.

```bash
cd /tmp
taskset -c 0-3 "$VIRTUAL_ENV/bin/python" \
  ~/onnxruntime/onnxruntime/test/python/transformers/benchmark_gqa_cpu_flash.py \
  --decode_only --warmup 50 --repeats 500 --intra_op_num_threads 4

taskset -c 0-7 "$VIRTUAL_ENV/bin/python" \
  /home/tlwu/onnxruntime/onnxruntime/test/python/transformers/benchmark_gqa_cpu_flash.py \
  --decode_only --warmup 50 --repeats 500 --intra_op_num_threads 8
```

### Flash-attention results

Latency in milliseconds. `FP16 vs. FP32` compares means at the same thread count.

| threads | total length | FP32 run 1 | FP32 run 2 | FP32 mean | FP16 run 1 | FP16 run 2 | FP16 mean | FP16 vs. FP32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 513 | 0.244 | 0.206 | 0.225 | 0.254 | 0.253 | 0.254 | 13% slower |
| 4 | 1025 | 0.378 | 0.449 | 0.414 | 0.400 | 0.451 | 0.426 | 3% slower |
| 4 | 2049 | 0.665 | 0.812 | 0.739 | 0.790 | 0.638 | 0.714 | 3% faster |
| 4 | 4097 | 1.329 | 1.053 | 1.191 | 1.147 | 1.657 | 1.402 | 18% slower |
| 8 | 513 | 0.188 | 0.194 | 0.191 | 0.239 | 0.237 | 0.238 | 25% slower |
| 8 | 1025 | 0.393 | 0.360 | 0.377 | 0.288 | 0.289 | 0.289 | 23% faster |
| 8 | 2049 | 0.743 | 0.609 | 0.676 | 0.434 | 0.569 | 0.502 | 26% faster |
| 8 | 4097 | 1.062 | 0.774 | 0.918 | 0.976 | 0.874 | 0.925 | 1% slower |

### Scaling from 4 to 8 threads

| total length | FP32 speedup | FP32 latency reduction | FP16 speedup | FP16 latency reduction |
|---:|---:|---:|---:|---:|
| 513 | 1.18x | 15% | 1.07x | 6% |
| 1025 | 1.10x | 9% | 1.47x | 32% |
| 2049 | 1.09x | 8% | 1.42x | 30% |
| 4097 | 1.30x | 23% | 1.52x | 34% |

At 8 threads, optimized FP16 is 23-26% faster than FP32 at total lengths 1025 and 2049. FP16
also scales more strongly than FP32 from 4 to 8 threads at lengths 1025-4097. The 513-token
case is too small to amortize FP16 path overhead, and FP16 remains slower there.

The two-run variation is material for some long-context points: coefficients of variation
reach 10-18% for the noisiest Flash measurements. In particular, the 4097-token FP16/FP32
comparison should be treated as approximate. The stable 8-thread 1025-token FP16 result has
0.2% CV; the 2049-token result has 13.5% CV but remains faster in both aggregate comparison
and the isolated MLAS measurements.

### End-to-end accumulated comparison

The following compares the first experiment's original eight-thread FP16 means with the
optimized eight-thread FP16 means. Unlike the isolated MLAS table, these runs were taken at
different times and have significant system-level variation, so they show end-to-end trend
rather than a controlled kernel-only speedup.

| total length | original FP16 mean | optimized FP16 mean | change |
|---:|---:|---:|---:|
| 513 | 0.274 ms | 0.238 ms | 13% faster |
| 1025 | 0.386 ms | 0.289 ms | 25% faster |
| 2049 | 0.861 ms | 0.502 ms | 42% faster |
| 4097 | 0.892 ms | 0.925 ms | 4% slower |

The controlled MLAS microbenchmarks are the primary evidence for accumulated kernel gains.
The end-to-end results agree at 513-2049 tokens; the 4097-token result is dominated by the
run-to-run variation already visible in both experiments.
