// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>

#include <vector>

#include "core/mlas/lib/mlasi.h"
#include "test/mlas/bench/bench_util.h"

namespace {

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

template <typename KernelFn>
void RunSiluUnfusedBenchmark(benchmark::State& state, KernelFn&& kernel) {
  const auto n = static_cast<size_t>(state.range(0));
  auto input = MakeInput(n, -20.0f, 20.0f);
  std::vector<float> output(n);

  kernel(input.data(), output.data(), n);

  for (auto _ : state) {
    kernel(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float) * 5));
}

template <typename KernelFn>
void RunGeluUnfusedBenchmark(benchmark::State& state, KernelFn&& kernel) {
  const auto n = static_cast<size_t>(state.range(0));
  auto input = MakeInput(n, -10.0f, 10.0f);
  std::vector<float> output(n);

  kernel(input.data(), output.data(), n);

  for (auto _ : state) {
    kernel(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float) * 7));
}

static void UnaryKernelArgs(benchmark::internal::Benchmark* b) {
  for (int n : {1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096, 16384, 65536, 262144}) {
    b->Arg(n);
  }
}

void BM_SiluDispatch(benchmark::State& state) {
  RunUnaryKernelBenchmark(state, MlasComputeSilu, -20.0f, 20.0f);
}

void BM_SiluUnfusedDispatch(benchmark::State& state) {
  RunSiluUnfusedBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        MlasComputeLogistic(input, output, n);
        MlasEltwiseMul<float>(input, output, output, n);
      });
}

void BM_GeluErfDispatchExact(benchmark::State& state) {
  RunUnaryKernelBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        MlasComputeGeluErf(input, output, n, MlasGeluErfModeExact);
      },
      -10.0f,
      10.0f);
}

void BM_GeluErfUnfusedExact(benchmark::State& state) {
  RunGeluUnfusedBenchmark(
      state,
      [](const float* input, float* output, size_t n) {
        constexpr float kInvSqrt2 = 0.7071067811865475244f;
        for (size_t i = 0; i < n; ++i) {
          output[i] = input[i] * kInvSqrt2;
        }

        MlasComputeErf(output, output, n);

        for (size_t i = 0; i < n; ++i) {
          output[i] = 0.5f * input[i] * (output[i] + 1.0f);
        }
      });
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

}  // namespace

BENCHMARK(BM_SiluDispatch)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_SiluUnfusedDispatch)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfDispatchExact)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfUnfusedExact)->Apply(UnaryKernelArgs)->UseRealTime();
BENCHMARK(BM_GeluErfDispatchMinimax)->Apply(UnaryKernelArgs)->UseRealTime();
