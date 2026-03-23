// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>

#include <vector>

#include "mlas.h"
#include "bench_util.h"
#include "core/mlas/lib/mlasi.h"

namespace {

// Compare fused MLAS unary activation paths against unfused baselines for
// SiLU and exact GELU(erf).

constexpr float kSiluMinValue = -20.0f;
constexpr float kSiluMaxValue = 20.0f;
constexpr float kGeluMinValue = -10.0f;
constexpr float kGeluMaxValue = 10.0f;
constexpr float kInvSqrt2 = 0.7071067811865475244f;
constexpr int64_t kFusedBytesPerElement = 2;
constexpr int64_t kSiluUnfusedBytesPerElement = 5;
constexpr int64_t kGeluUnfusedBytesPerElement = 7;

struct DispatchedUnaryPathInfo {
  int64_t bytes_per_element;
  const char* label;
};

DispatchedUnaryPathInfo GetSiluDispatchPathInfo() {
#if defined(MLAS_TARGET_AMD64)
  if (GetMlasPlatform().SiluKernelRoutine == MlasSiluKernelAvx512F) {
    return {kFusedBytesPerElement, "avx512_fused"};
  }
#endif

  // The current non-AVX512 dispatch target falls back to the generic path,
  // which materializes the logistic result before the final multiply.
  return {kSiluUnfusedBytesPerElement, "generic_fallback"};
}

DispatchedUnaryPathInfo GetGeluErfDispatchPathInfo() {
#if defined(MLAS_TARGET_AMD64)
  if (GetMlasPlatform().GeluErfKernelRoutine == MlasGeluErfKernelAvx512F) {
    return {kFusedBytesPerElement, "avx512_fused"};
  }
#endif

  // The current non-AVX512 dispatch target falls back to the generic exact
  // GELU(erf) implementation, which uses separate scale, erf, and final passes.
  return {kGeluUnfusedBytesPerElement, "generic_fallback"};
}

std::vector<float> MakeInput(size_t n, float min_value, float max_value) {
  auto data = RandomVectorUniform<float>(n, min_value, max_value);

  if (!data.empty()) {
    data[0] = 0.0f;
  }
  if (data.size() > 1) {
    data[1] = -0.0f;
  }
  if (data.size() > 2) {
    data[2] = -1.0f;
  }
  if (data.size() > 3) {
    data[3] = 1.0f;
  }

  return data;
}

template <typename KernelFn>
void RunDispatchedUnaryBenchmark(benchmark::State& state,
                                 KernelFn&& kernel,
                                 float min_value,
                                 float max_value,
                                 DispatchedUnaryPathInfo path_info) {
  const auto n = static_cast<size_t>(state.range(0));
  auto input = MakeInput(n, min_value, max_value);
  std::vector<float> output(n);

  state.SetLabel(path_info.label);

  kernel(input.data(), output.data(), n);

  for (auto _ : state) {
    kernel(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  const int64_t bytes_per_iteration = static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(float)) * path_info.bytes_per_element;
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_per_iteration);
}

template <typename KernelFn>
void RunUnfusedUnaryBenchmark(benchmark::State& state,
                              KernelFn&& kernel,
                              float min_value,
                              float max_value,
                              int64_t bytes_per_element) {
  const auto n = static_cast<size_t>(state.range(0));
  auto input = MakeInput(n, min_value, max_value);
  std::vector<float> output(n);

  kernel(input.data(), output.data(), n);

  for (auto _ : state) {
    kernel(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  const int64_t bytes_per_iteration = static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(float)) * bytes_per_element;
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_per_iteration);
}

static void UnaryKernelArgs(benchmark::internal::Benchmark* b) {
  for (int n : {1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096, 16384, 65536, 262144}) {
    b->Arg(n);
  }
}

void BM_SiluDispatch(benchmark::State& state) {
  // Fused MLAS SiLU entry point. On supported platforms this may dispatch to a
  // specialized implementation that combines the activation into a single
  // kernel instead of exposing intermediate results.
  RunDispatchedUnaryBenchmark(state, MlasComputeSilu, kSiluMinValue, kSiluMaxValue, GetSiluDispatchPathInfo());
}

void BM_SiluUnfusedDispatch(benchmark::State& state) {
  // Unfused SiLU baseline: compute logistic(x) first and then multiply by x in
  // a separate elementwise pass.
  RunUnfusedUnaryBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        MlasComputeLogistic(input, output, n);
        MlasEltwiseMul<float>(input, output, output, n);
      },
      kSiluMinValue,
      kSiluMaxValue,
      kSiluUnfusedBytesPerElement);
}

void BM_GeluErfDispatchExact(benchmark::State& state) {
  // Fused MLAS GELU(erf) entry point using the exact erf-based formulation.
  // On AMD64 this goes through the platform dispatch layer and may select an
  // architecture-specific implementation.
  RunDispatchedUnaryBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        MlasComputeGeluErf(input, output, n);
      },
      kGeluMinValue,
      kGeluMaxValue,
      GetGeluErfDispatchPathInfo());
}

void BM_GeluErfUnfusedExact(benchmark::State& state) {
  // Unfused exact GELU(erf) baseline: scale by 1/sqrt(2), run erf, then apply the
  // final 0.5 * x * (erf(x / sqrt(2)) + 1) transform in a separate pass.
  RunUnfusedUnaryBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        for (size_t i = 0; i < n; ++i) {
          output[i] = input[i] * kInvSqrt2;
        }

        MlasComputeErf(output, output, n);

        for (size_t i = 0; i < n; ++i) {
          output[i] = 0.5f * input[i] * (output[i] + 1.0f);
        }
      },
      kGeluMinValue,
      kGeluMaxValue,
      kGeluUnfusedBytesPerElement);
}

}  // namespace

BENCHMARK(BM_SiluDispatch)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_SiluUnfusedDispatch)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfDispatchExact)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfUnfusedExact)->Apply(UnaryKernelArgs)->UseRealTime();
