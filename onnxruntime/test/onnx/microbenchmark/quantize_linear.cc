#include "common.h"

#include <benchmark/benchmark.h>
#include "core/util/math_cpuonly.h"
#include "core/mlas/lib/mlasi.h"

using namespace onnxruntime;

static void BM_QuantizeLinearSSE2(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  for (auto _ : state) {
    MlasQuantizeLinearU8Kernal(data, output, batch_size, 2.f / 512.f, 1);
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_QuantizeLinearSSE2)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(64)
    ->Arg(80)
    ->Arg(100)
    ->Arg(128)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);

static void BM_QuantizeLinearAVX512(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  for (auto _ : state) {
    MlasQuantizeLinearU8KernelAvx512F(data, output, batch_size, 2.f / 512.f, 1);
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_QuantizeLinearAVX512)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(64)
    ->Arg(80)
    ->Arg(100)
    ->Arg(128)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);
