#include "common.h"

#include <benchmark/benchmark.h>
#include "core/util/qmath.h"
#include "core/util/thread_utils.h"

static void BenchSize(benchmark::internal::Benchmark* b) {
  for (int size : {80000, 160000, 320000, 640000, 1280000}) {
    for (int threads : {2, 4, 6, 8}) {
      b->Args({size, threads});
    }
  }
}

// qmath.h GetQuantizationParameter
struct quant_params {
  float scale;
  uint8_t zero;
};

static quant_params GetQuantParams(const float* data, const int64_t data_size, onnxruntime::concurrency::ThreadPool* tp) {
  quant_params params;
  onnxruntime::GetQuantizationParameter<uint8_t>(data, data_size, params.scale, params.zero, tp);
  return params;
}

static void BM_GetQuantParams(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  const int64_t threads = state.range(1);

  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    benchmark::DoNotOptimize(GetQuantParams(data, batch_size, tp.get()));
  }
  aligned_free(data);
}

BENCHMARK(BM_GetQuantParams)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Apply(BenchSize);

static void BM_Quantize(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  const int64_t threads = state.range(1);
  float* a_data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  uint8_t* a_data_quant = static_cast<uint8_t*>(aligned_alloc(sizeof(uint8_t) * batch_size, 64));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  float scale = 1.23456f;
  uint8_t zero = 129;

  for (auto _ : state) {
    benchmark::DoNotOptimize(a_data_quant);
    onnxruntime::ParQuantizeLinear(a_data, a_data_quant, batch_size, scale, zero, tp.get());
    benchmark::ClobberMemory();
  }
  aligned_free(a_data_quant);
  aligned_free(a_data);
}

BENCHMARK(BM_Quantize)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Apply(BenchSize);
