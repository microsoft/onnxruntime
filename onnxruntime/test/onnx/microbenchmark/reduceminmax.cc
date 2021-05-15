#include "common.h"

#include <benchmark/benchmark.h>
#include "core/mlas/lib/mlasi.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

// vanilla implementation of FindMinMax
static void BM_FindMinMaxPlainLoop(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    for (size_t i = 0; i != batch_size; ++i) {
      if (min > data[i]) {
        min = data[i];
      }
      if (max < data[i]) {
        max = data[i];
      }
    }
  }

  // To prevent to optimize out min and max
  data[0] = min * max;
  aligned_free(data);
}

BENCHMARK(BM_FindMinMaxPlainLoop)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(160000);

// Eigen implementation of FindMinMax
static void BM_FindMinMaxEigen(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  for (auto _ : state) {
    onnxruntime::ConstEigenVectorMap<float>(data, batch_size).minCoeff();
    onnxruntime::ConstEigenVectorMap<float>(data, batch_size).maxCoeff();
  }
  aligned_free(data);
}

BENCHMARK(BM_FindMinMaxEigen)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(160000);

// MLAS sse2 implementation
static void BM_FindMinMaxMlasSSE2(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    MlasReduceMinimumMaximumF32Kernel(data, &min, &max, batch_size);
  }
  aligned_free(data);
}

BENCHMARK(BM_FindMinMaxMlasSSE2)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(160000);

// MLAS avx implementation
static void BM_FindMinMaxMlasAvx(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    MlasReduceMinimumMaximumF32KernelAvx(data, &min, &max, batch_size);
  }
  aligned_free(data);
}

BENCHMARK(BM_FindMinMaxMlasAvx)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(160000);

