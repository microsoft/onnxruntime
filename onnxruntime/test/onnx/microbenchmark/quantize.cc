#include "common.h"

#include <benchmark/benchmark.h>
#include "core/util/qmath.h"


// qmath.h GetQuantizationParameter
struct quant_params {
  float scale;
  uint8_t zero;
};

static quant_params GetQuantParams(const float* data, const int64_t data_size) {
  quant_params params;
  onnxruntime::GetQuantizationParameter<uint8_t>(data, data_size, params.scale, params.zero);
  return params;
}

static void BM_GetQuantParams(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(GetQuantParams(data, batch_size));
  }
  aligned_free(data);
}

BENCHMARK(BM_GetQuantParams)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Quantize(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  float* a_data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  uint8_t* a_data_quant = static_cast<uint8_t*>(aligned_alloc(sizeof(uint8_t) * batch_size, 64));
  float scale = 1.23456f;
  uint8_t zero = 129;

  for (auto _ : state) {
    benchmark::DoNotOptimize(a_data_quant);
    MlasQuantizeLinear(a_data, a_data_quant, batch_size, scale, zero);
    benchmark::ClobberMemory();
  }
  aligned_free(a_data_quant);
  aligned_free(a_data);
}

BENCHMARK(BM_Quantize)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);
