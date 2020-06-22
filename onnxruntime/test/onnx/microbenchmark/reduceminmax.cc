#include "common.h"

#include <benchmark/benchmark.h>
#include "core/mlas/lib/mlasi.h"
#include "core/util/math_cpuonly.h"

//naive implementation of FindMinMax
static void BM_FindMinMaxPlainLoop(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  for (auto _ : state) {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i != batch_size; ++i) {
      if (min > data[i]) min = data[i];
      if (max < data[i]) max = data[i];
    }
    data[0] = min * max;
  }
  aligned_free(data);
  aligned_free(output);
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
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    min = onnxruntime::ConstEigenVectorMap<float>(data, batch_size).minCoeff();
    max = onnxruntime::ConstEigenVectorMap<float>(data, batch_size).maxCoeff();
  }
  data[0] = min * max;
  aligned_free(data);
  aligned_free(output);
}

//Use eigen and mlas to implement Gelu, single thread
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

// Implementation with Mlas generic kernel
static void BM_FindMinMaxMlasGeneric(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    MlasReduceMinimumMaximumF32Kernel(data, &min, &max, batch_size);
  }
  aligned_free(data);
  aligned_free(output);
}

//Use eigen and mlas to implement Gelu, single thread
BENCHMARK(BM_FindMinMaxMlasGeneric)
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

// Implementation with Mlas avx kernel
static void BM_FindMinMaxMlasAvx(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    MlasReduceMinimumMaximumF32KernelAvx(data, &min, &max, batch_size);
  }
  aligned_free(data);
  aligned_free(output);
}

//Use eigen and mlas to implement Gelu, single thread
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

// Implementation with Mlas avx kernel
static void BM_FindMaxMlasAvx(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  float max = std::numeric_limits<float>::lowest();
  for (auto _ : state) {
    max = MlasReduceMaximumF32KernelAvx(data, batch_size);
  }
  data[0] = max;
  aligned_free(data);
  aligned_free(output);
}

//Use eigen and mlas to implement Gelu, single thread
BENCHMARK(BM_FindMaxMlasAvx)
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