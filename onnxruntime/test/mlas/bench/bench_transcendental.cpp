// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>

#include <vector>

#include "core/mlas/lib/mlasi.h"
#include "test/mlas/bench/bench_util.h"

namespace {

bool IsAvx512Available() {
#if defined(MLAS_TARGET_AMD64)
  return GetMlasPlatform().Avx512Supported_;
#else
  return false;
#endif
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
void RunUnaryKernelBenchmark(benchmark::State& state,
                             KernelFn&& kernel,
                             float min_value,
                             float max_value) {
  const auto n = static_cast<size_t>(state.range(0));
  auto input = MakeInput(n, min_value, max_value);
  std::vector<float> output(n);

  kernel(input.data(), output.data(), n);

  for (auto _ : state) {
    kernel(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float) * 2));
}

static void UnaryKernelArgs(benchmark::internal::Benchmark* b) {
  for (int n : {1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096, 16384, 65536, 262144}) {
    b->Arg(n);
  }
}

void BM_SiluGeneric(benchmark::State& state) {
  RunUnaryKernelBenchmark(state, MlasSiluKernel, -20.0f, 20.0f);
}

void BM_SiluDispatch(benchmark::State& state) {
  RunUnaryKernelBenchmark(state, MlasComputeSilu, -20.0f, 20.0f);
}

#if defined(MLAS_TARGET_AMD64)

void BM_SiluAvx512F(benchmark::State& state) {
  if (!IsAvx512Available()) {
    state.SkipWithError("AVX512 is not available on this machine.");
    return;
  }

  RunUnaryKernelBenchmark(state, MlasSiluKernelAvx512F, -20.0f, 20.0f);
}

void BM_GeluErfGeneric(benchmark::State& state) {
  RunUnaryKernelBenchmark(state, MlasGeluKernel, -10.0f, 10.0f);
}

void BM_GeluErfDispatchExact(benchmark::State& state) {
  RunUnaryKernelBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        MlasComputeGeluErf(input, output, n, MlasGeluErfModeDefault);
      },
      -10.0f,
      10.0f);
}

void BM_GeluErfAvx512FExact(benchmark::State& state) {
  if (!IsAvx512Available()) {
    state.SkipWithError("AVX512 is not available on this machine.");
    return;
  }

  RunUnaryKernelBenchmark(state, MlasGeluKernelAvx512F, -10.0f, 10.0f);
}

void BM_GeluErfDispatchMinimax(benchmark::State& state) {
  if (GetMlasPlatform().GeluErfMinimaxKernelRoutine == nullptr) {
    state.SkipWithError("GELU erf minimax kernel is not available on this machine.");
    return;
  }

  RunUnaryKernelBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        MlasComputeGeluErf(input, output, n, MlasGeluErfModeMinimaxApproximation);
      },
      -10.0f,
      10.0f);
}

void BM_GeluErfAvx512FMinimax(benchmark::State& state) {
  if (GetMlasPlatform().GeluErfMinimaxKernelRoutine == nullptr) {
    state.SkipWithError("GELU erf minimax kernel is not available on this machine.");
    return;
  }

  RunUnaryKernelBenchmark(state, MlasGeluKernelAvx512FMinimaxApprox, -10.0f, 10.0f);
}

#endif  // defined(MLAS_TARGET_AMD64)

}  // namespace

BENCHMARK(BM_SiluGeneric)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_SiluDispatch)->Apply(UnaryKernelArgs)->UseRealTime();

#if defined(MLAS_TARGET_AMD64)
BENCHMARK(BM_SiluAvx512F)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfGeneric)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfDispatchExact)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfAvx512FExact)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfDispatchMinimax)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfAvx512FMinimax)->Apply(UnaryKernelArgs)->UseRealTime();
#endif  // defined(MLAS_TARGET_AMD64)
