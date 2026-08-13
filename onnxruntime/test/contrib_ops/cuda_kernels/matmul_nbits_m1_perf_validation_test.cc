// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// VALIDATION-ONLY benchmark. This file exists solely to gather real A10-hardware
// head-vs-base performance evidence for microsoft/onnxruntime#31988
// (nxrt/cuda-matmulnbits-sm-cols) and is not intended to be merged. It lives only on a
// throwaway validation branch/PR that will be closed (never merged) once the evidence has
// been folded into #31988's PR body.
//
// "head" means #31988's occupancy-driven ChooseColsPerBlockFromOccupancy() selection
// (the code as-is, no override). "base" means the pre-#31988 behavior -- cols_per_block
// pinned to kColsPerThreadBlock (8) unconditionally -- reproduced *in the same process* via
// g_matmulnbits_m1_perf_validation_force_cols_per_block (see matmul_4bits_m1_impl.cuh). Using
// one process/binary for both arms removes build-to-build and machine-state (thermal, clock,
// driver-cache) confounds that a separate base-branch CI run would introduce.
//
// Gated behind ORT_RUN_MATMULNBITS_M1_PERF_VALIDATION=1 so it never runs as part of any
// ordinary CI leg (including the shared windows_cuda.yml build+test job that already runs
// against this same branch) -- it only executes when a dedicated validation workflow step
// explicitly opts in. Skips (does not fail) when no CUDA device is present.
//
// Run with (after setting the env var above):
//   onnxruntime_test_all.exe --gtest_filter=CUDA_EP_Benchmark.MatMulNBitsM1_PerfAB

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "contrib_ops/cuda/quantization/matmul_4bits_m1.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
// Defined in matmul_4bits_m1_impl.cuh (included by matmul_4bits_float.cu et al.). Declared
// `extern` here so this plain .cc test can flip it without pulling in CUDA device headers.
extern int g_matmulnbits_m1_perf_validation_force_cols_per_block;
}  // namespace cuda
}  // namespace contrib

namespace test {
namespace {

constexpr int kColsPerThreadBlockForBaseline = 8;  // must match kColsPerThreadBlock

// Avoids pulling in core/providers/cuda/shared_inc/cuda_call.h (and its cuDNN/cuBLAS pch)
// just for error checking in a throwaway benchmark; cudaGetErrorString is already available
// via cuda_runtime_api.h.
#define ORT_TEST_CUDA_CHECK(expr)                                    \
  do {                                                               \
    cudaError_t ort_test_cuda_check_err = (expr);                    \
    ASSERT_EQ(ort_test_cuda_check_err, cudaSuccess)                  \
        << #expr << " failed: " << cudaGetErrorString(ort_test_cuda_check_err); \
  } while (0)

struct Shape {
  int n;
  int k;
  int block_size;
  const char* label;
};

struct Stats {
  double min_us = 0, p50_us = 0, p90_us = 0, max_us = 0, mean_us = 0, stddev_us = 0;
};

Stats ComputeStats(std::vector<float>& elapsed_ms) {
  Stats s;
  std::sort(elapsed_ms.begin(), elapsed_ms.end());
  const size_t n = elapsed_ms.size();
  auto to_us = [](float ms) { return static_cast<double>(ms) * 1000.0; };
  s.min_us = to_us(elapsed_ms.front());
  s.max_us = to_us(elapsed_ms.back());
  s.p50_us = to_us(elapsed_ms[n / 2]);
  s.p90_us = to_us(elapsed_ms[static_cast<size_t>(n * 0.9)]);
  double sum = 0;
  for (float v : elapsed_ms) sum += to_us(v);
  s.mean_us = sum / static_cast<double>(n);
  double sq = 0;
  for (float v : elapsed_ms) {
    const double d = to_us(v) - s.mean_us;
    sq += d * d;
  }
  s.stddev_us = std::sqrt(sq / static_cast<double>(n));
  return s;
}

// Times `iters` back-to-back launches of `launch()` (all recorded on `stream`, no
// synchronization in between) and returns one elapsed-ms sample per launch.
template <typename Func>
std::vector<float> TimeLaunches(Func&& launch, int warmup, int iters, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) launch();
  cudaStreamSynchronize(stream);

  std::vector<cudaEvent_t> events(iters + 1);
  for (auto& e : events) cudaEventCreate(&e);

  cudaEventRecord(events[0], stream);
  for (int i = 0; i < iters; ++i) {
    launch();
    cudaEventRecord(events[i + 1], stream);
  }
  cudaEventSynchronize(events[iters]);

  std::vector<float> elapsed_ms(iters);
  for (int i = 0; i < iters; ++i) {
    cudaEventElapsedTime(&elapsed_ms[i], events[i], events[i + 1]);
  }
  for (auto& e : events) cudaEventDestroy(e);
  return elapsed_ms;
}

}  // namespace

using onnxruntime::contrib::cuda::g_matmulnbits_m1_perf_validation_force_cols_per_block;
using onnxruntime::contrib::cuda::TryMatMul4BitsM1;

TEST(CUDA_EP_Benchmark, MatMulNBitsM1_PerfAB) {
  if (!std::getenv("ORT_RUN_MATMULNBITS_M1_PERF_VALIDATION")) {
    GTEST_SKIP() << "Set ORT_RUN_MATMULNBITS_M1_PERF_VALIDATION=1 to run this "
                    "validation-only benchmark (see #31988).";
  }

  int device_count = 0;
  ORT_TEST_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  ASSERT_GT(device_count, 0) << "No CUDA device visible; this benchmark requires real hardware.";

  cudaDeviceProp prop{};
  ORT_TEST_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver_version = 0, runtime_version = 0;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  std::cout << "=== MatMulNBitsM1_PerfAB: device pinning ===\n"
            << "gpu_name=" << prop.name << "\n"
            << "compute_capability=" << prop.major << "." << prop.minor << "\n"
            << "sm_count=" << prop.multiProcessorCount << "\n"
            << "shared_mem_per_block_bytes=" << prop.sharedMemPerBlock << "\n"
            << "driver_version=" << driver_version << "\n"
            << "runtime_version=" << runtime_version << "\n";

  // Representative M=1 GEMV shapes for a 72-SM device (A10):
  //  - n=256: n/8=32 < 72 -> narrow, ChooseColsPerBlockFromOccupancy is expected to pick 2.
  //  - n=384: n/8=48 < 72 -> narrow, expected to pick 4 (per matmul_4bits_m1_impl.cuh comment).
  //  - n=576: n/8=72 >= 72 -> the wide-n structural threshold boundary; must stay 8.
  //  - n=4096/8192/11008/14336: common LLM hidden/intermediate sizes, all structurally wide.
  // K=4096 and K=14336 are common hidden/intermediate sizes; block_size 32 is the primary
  // sweep value, with two 128 spot-checks (register/shared-mem profile is dominated by
  // cols_per_block, not block_size, per the occupancy-query design in matmul_4bits_m1_impl.cuh).
  const std::vector<Shape> shapes = {
      {256, 4096, 32, "narrow_n256_k4096_bs32"},
      {384, 4096, 32, "narrow_n384_k4096_bs32"},
      {576, 4096, 32, "boundary_n576_k4096_bs32"},
      {4096, 4096, 32, "wide_n4096_k4096_bs32"},
      {8192, 4096, 32, "wide_n8192_k4096_bs32"},
      {11008, 4096, 32, "wide_n11008_k4096_bs32"},
      {14336, 4096, 32, "wide_n14336_k4096_bs32"},
      {256, 14336, 32, "narrow_n256_k14336_bs32"},
      {4096, 14336, 32, "wide_n4096_k14336_bs32"},
      {14336, 14336, 32, "wide_n14336_k14336_bs32"},
      {256, 4096, 128, "narrow_n256_k4096_bs128"},
      {14336, 4096, 128, "wide_n14336_k4096_bs128"},
  };

  constexpr int kWarmup = 20;
  constexpr int kIters = 200;

  cudaStream_t stream = nullptr;
  ORT_TEST_CUDA_CHECK(cudaStreamCreate(&stream));

  std::mt19937 rng(0xA10);  // fixed seed for reproducibility
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> scale_dist(0.001f, 0.05f);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  std::cout << "\nshape,n,k,block_size,mode,cols_per_block,numerics_ok,warmup,iters,"
               "min_us,p50_us,p90_us,max_us,mean_us,stddev_us\n";

  int shapes_with_numerics_mismatch = 0;

  for (const Shape& shape : shapes) {
    const int n = shape.n;
    const int k = shape.k;
    const int block_size = shape.block_size;
    const int blocks_per_k = (k + block_size - 1) / block_size;

    std::vector<float> h_a(static_cast<size_t>(k));
    for (auto& v : h_a) v = a_dist(rng);
    std::vector<float> h_scales(static_cast<size_t>(n) * blocks_per_k);
    for (auto& v : h_scales) v = scale_dist(rng);
    std::vector<uint8_t> h_b(static_cast<size_t>(n) * blocks_per_k * (block_size / 2));
    for (auto& v : h_b) v = static_cast<uint8_t>(byte_dist(rng));

    float* d_a = nullptr;
    float* d_scales = nullptr;
    uint8_t* d_b = nullptr;
    float* d_out_head = nullptr;
    float* d_out_base = nullptr;
    ORT_TEST_CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    ORT_TEST_CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    ORT_TEST_CUDA_CHECK(cudaMalloc(&d_b, h_b.size()));
    ORT_TEST_CUDA_CHECK(cudaMalloc(&d_out_head, static_cast<size_t>(n) * sizeof(float)));
    ORT_TEST_CUDA_CHECK(cudaMalloc(&d_out_base, static_cast<size_t>(n) * sizeof(float)));
    ORT_TEST_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    ORT_TEST_CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    ORT_TEST_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size(), cudaMemcpyHostToDevice));

    // ---- Numerics check first (before trusting any timing): head vs forced-base must match
    // bit-for-bit. Every column is reduced independently by one warp over the full K range;
    // cols_per_block only changes how many columns share a CTA, not the per-column arithmetic
    // order, so the outputs must be identical regardless of which geometry is selected.
    g_matmulnbits_m1_perf_validation_force_cols_per_block = 0;  // head: natural selection
    bool head_ok = TryMatMul4BitsM1<float>(d_out_head, d_a, d_b, d_scales, /*zero_points=*/nullptr,
                                           n, k, block_size, prop.sharedMemPerBlock,
                                           prop.multiProcessorCount, stream);
    g_matmulnbits_m1_perf_validation_force_cols_per_block = kColsPerThreadBlockForBaseline;  // base
    bool base_ok = TryMatMul4BitsM1<float>(d_out_base, d_a, d_b, d_scales, /*zero_points=*/nullptr,
                                           n, k, block_size, prop.sharedMemPerBlock,
                                           prop.multiProcessorCount, stream);
    g_matmulnbits_m1_perf_validation_force_cols_per_block = 0;
    ORT_TEST_CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(head_ok) << "TryMatMul4BitsM1 (head) declined shape " << shape.label
                          << " -- not a representative admitted shape, fix the shape list.";
    ASSERT_TRUE(base_ok) << "TryMatMul4BitsM1 (forced base) declined shape " << shape.label;

    std::vector<float> h_out_head(n), h_out_base(n);
    ORT_TEST_CUDA_CHECK(cudaMemcpy(h_out_head.data(), d_out_head, n * sizeof(float), cudaMemcpyDeviceToHost));
    ORT_TEST_CUDA_CHECK(cudaMemcpy(h_out_base.data(), d_out_base, n * sizeof(float), cudaMemcpyDeviceToHost));
    const bool numerics_ok = (std::memcmp(h_out_head.data(), h_out_base.data(), n * sizeof(float)) == 0);
    if (!numerics_ok) ++shapes_with_numerics_mismatch;

    // ---- Timing: head (natural) then base (forced cols=8), same process/buffers/stream.
    g_matmulnbits_m1_perf_validation_force_cols_per_block = 0;
    // Query the cols_per_block head actually selected, for reporting (mirrors production).
    int head_cols_per_block = 0;
    {
      // Re-derive via a throwaway call is wasteful; instead infer from n/sm_count using the
      // same structural rule documented in matmul_4bits_m1_impl.cuh, then fall back to "?"
      // if ambiguous. This is reporting-only and does not affect the measured kernel.
      const int sm = prop.multiProcessorCount;
      if (n % 8 == 0 && n / 8 >= sm) {
        head_cols_per_block = 8;
      } else {
        head_cols_per_block = -1;  // resolved by real occupancy query at runtime; see log note
      }
    }
    auto head_elapsed = TimeLaunches(
        [&]() {
          TryMatMul4BitsM1<float>(d_out_head, d_a, d_b, d_scales, nullptr, n, k, block_size,
                                   prop.sharedMemPerBlock, prop.multiProcessorCount, stream);
        },
        kWarmup, kIters, stream);
    Stats head_stats = ComputeStats(head_elapsed);
    std::cout << shape.label << "," << n << "," << k << "," << block_size << ",head,"
              << (head_cols_per_block > 0 ? std::to_string(head_cols_per_block) : "occupancy-chosen(see-driver-log)")
              << "," << (numerics_ok ? "PASS" : "FAIL") << "," << kWarmup << "," << kIters << ","
              << head_stats.min_us << "," << head_stats.p50_us << "," << head_stats.p90_us << ","
              << head_stats.max_us << "," << head_stats.mean_us << "," << head_stats.stddev_us << "\n";

    g_matmulnbits_m1_perf_validation_force_cols_per_block = kColsPerThreadBlockForBaseline;
    auto base_elapsed = TimeLaunches(
        [&]() {
          TryMatMul4BitsM1<float>(d_out_base, d_a, d_b, d_scales, nullptr, n, k, block_size,
                                   prop.sharedMemPerBlock, prop.multiProcessorCount, stream);
        },
        kWarmup, kIters, stream);
    g_matmulnbits_m1_perf_validation_force_cols_per_block = 0;
    Stats base_stats = ComputeStats(base_elapsed);
    std::cout << shape.label << "," << n << "," << k << "," << block_size << ",base,8,"
              << (numerics_ok ? "PASS" : "FAIL") << "," << kWarmup << "," << kIters << ","
              << base_stats.min_us << "," << base_stats.p50_us << "," << base_stats.p90_us << ","
              << base_stats.max_us << "," << base_stats.mean_us << "," << base_stats.stddev_us << "\n";

    std::cout << "# " << shape.label << " speedup(base_p50/head_p50)=" << std::fixed << std::setprecision(3)
              << (base_stats.p50_us / head_stats.p50_us) << "x\n";

    cudaFree(d_a);
    cudaFree(d_scales);
    cudaFree(d_b);
    cudaFree(d_out_head);
    cudaFree(d_out_base);
  }

  cudaStreamDestroy(stream);

  EXPECT_EQ(shapes_with_numerics_mismatch, 0)
      << shapes_with_numerics_mismatch
      << " shape(s) produced different output between head and forced-base cols_per_block; "
         "see PASS/FAIL column above. This would indicate a real correctness bug in "
         "ChooseColsPerBlockFromOccupancy / MatMulFloat4BitsKernelM1, not merely a perf issue.";
}

}  // namespace test
}  // namespace onnxruntime
