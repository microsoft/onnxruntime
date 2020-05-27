#include "common.h"

#include <benchmark/benchmark.h>
#include <core/util/math_cpuonly.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/util/thread_utils.h>
#include <core/providers/cpu/nn/pool_functors.h>

#include <mlas.h>

using namespace onnxruntime;

//naive implementation of QuantizeLinear
static void BM_QuantizeLinearSingleThreadPlainLoop(benchmark::State& state) {
  const float scale = 0.1f;
  const uint8_t zero_point = 127;

  const size_t batch_size = static_cast<size_t>(state.range(0));
  uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  for (auto _ : state) {
    for (size_t i = 0; i != batch_size; ++i) {
      float x = data[i];
      output[i] = static_cast<int8_t>(Clamp<int>(static_cast<int>(x / scale + zero_point), 0, 255));
    }
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_QuantizeLinearSingleThreadPlainLoop)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000);

//Use mlas to implement QuantizeLinear
static void BM_QuantizeLinearSingleThreadMlas(benchmark::State& state) {
  const float scale = 0.1f;
  const uint8_t zero_point = 127;

  const size_t batch_size = static_cast<size_t>(state.range(0));
  uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  for (auto _ : state) {
    MlasQuantizeLinear(data, output, batch_size, scale, zero_point);
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_QuantizeLinearSingleThreadMlas)
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
    ->Arg(1572864);

//Use ParallelFor to implement Gelu, single thread
static void BM_QuantizeLinearParallelFor(benchmark::State& state) {
  const float scale = 0.1f;
  const uint8_t zero_point = 127;

  const size_t batch_size = static_cast<size_t>(state.range(0));
  uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  for (auto _ : state) {
    concurrency::ThreadPool::TryParallelFor(tp.get(), batch_size, 10.0 /*cost*/, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      MlasQuantizeLinear(data + begin, output + begin, end - begin, scale, zero_point);
    });
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_QuantizeLinearParallelFor)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(1000)
    ->Arg(2500)
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(98304)
    ->Arg(1572864);

static void BM_DequantizeLinearSingleThread(benchmark::State& state) {
  const float sc = 0.1f;
  const uint8_t zp = 127;

  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  uint8_t* data = GenerateArrayWithRandomValue<uint8_t>(batch_size, 0, 255);
  for (auto _ : state) {
    for (size_t idx = 0; idx < batch_size; idx++) {
      output[idx] = static_cast<float>(static_cast<int32_t>(data[idx]) - zp) * sc;
    }
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_DequantizeLinearSingleThread)
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
    ->Arg(1572864);

static void BM_DequantizeLinearParallelFor(benchmark::State& state) {
  const float sc = 0.1f;
  const uint8_t zp = 127;

  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  uint8_t* data = GenerateArrayWithRandomValue<uint8_t>(batch_size, 0, 255);

  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    concurrency::ThreadPool::TryParallelFor(tp.get(), batch_size, 10.0 /*cost*/, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      const uint8_t* input_tmp = data + begin;
      float* output_tmp = output + begin;
      for (; output_tmp != output + end;) {
        *output_tmp++ = static_cast<float>(static_cast<int32_t>(*input_tmp++) - zp) * sc;
      }
    });
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_DequantizeLinearParallelFor)
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
    ->Arg(1572864);
