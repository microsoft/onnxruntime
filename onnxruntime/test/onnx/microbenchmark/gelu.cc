#include "common.h"

#include <benchmark/benchmark.h>
#include <core/util/math_cpuonly.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/util/thread_utils.h>
#include <core/providers/cpu/nn/pool_functors.h>
#include <mlas.h>

using namespace onnxruntime;

//naive implementation of Gelu
static void BM_GeluSingleThreadPlainLoop(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  for (auto _ : state) {
    for (size_t i = 0; i != batch_size; ++i) {
      float x = data[i];
      output[i] = x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
    }
  }
  aligned_free(data);
  aligned_free(output);
}

//Use eigen and mlas to implement Gelu, single thread
BENCHMARK(BM_GeluSingleThreadPlainLoop)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000);

static void BM_GeluSingleThreadMlas(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  for (auto _ : state) {
    onnxruntime::ConstEigenVectorArrayMap<float> xm(data, batch_size);
    onnxruntime::EigenVectorArrayMap<float> ym(output, batch_size);
    ym = xm * static_cast<float>(M_SQRT1_2);
    MlasComputeErf(output, output, batch_size);
    ym = xm * 0.5f * (ym + 1.0f);
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_GeluSingleThreadMlas)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000);

//Use ParallelFor to implement Gelu, single thread
static void BM_GeluParallelFor(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const int cost = static_cast<int>(state.range(1));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  for (auto _ : state) {
    tp->ParallelFor(batch_size, cost, [data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      float* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<float> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<float> ym(output_ptr, len);
      ym = xm * static_cast<float>(M_SQRT1_2);
      MlasComputeErf(output_ptr, output_ptr, len);
      ym = xm * 0.5f * (ym + 1.0f);
    });
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_GeluParallelFor)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({1000, 1})
    ->Args({1000, 5})
    ->Args({1000, 10})
    ->Args({1000, 40})
    ->Args({2500, 1})
    ->Args({2500, 5})
    ->Args({2500, 10})
    ->Args({2500, 40})
    ->Args({2500, 80})
    ->Args({2500, 160})
    ->Args({5000, 1})
    ->Args({5000, 5})
    ->Args({5000, 10})
    ->Args({5000, 40})
    ->Args({5000, 80})
    ->Args({5000, 160})
    ->Args({10000, 1})
    ->Args({10000, 5})
    ->Args({10000, 10})
    ->Args({10000, 40})
    ->Args({20000, 1})
    ->Args({20000, 5})
    ->Args({20000, 10})
    ->Args({20000, 40})
    ->Args({40000, 1})
    ->Args({40000, 5})
    ->Args({40000, 10})
    ->Args({40000, 40})
    ->Args({98304, 1})
    ->Args({98304, 5})
    ->Args({98304, 10})
    ->Args({98304, 40})
    ->Args({1572864, 1})
    ->Args({1572864, 5})
    ->Args({1572864, 10})
    ->Args({1572864, 40});

static void BM_ScaledTanhParallelFor(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const int cost = static_cast<int>(state.range(1));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  const float alpha_ = 0.3f;
  const float beta_ = 0.6f;
  for (auto _ : state) {
    tp->ParallelFor(batch_size, cost, [alpha_, beta_, data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      float* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<float> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<float> ym(output_ptr, len);
      ym = (xm >= 0).select(xm, alpha_ * (xm.exp() - 1));
    });
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_ScaledTanhParallelFor)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({1000, 1})
    ->Args({1000, 5})
    ->Args({1000, 10})
    ->Args({1000, 40})
    ->Args({2500, 1})
    ->Args({2500, 5})
    ->Args({2500, 10})
    ->Args({2500, 40})
    ->Args({2500, 80})
    ->Args({2500, 160})
    ->Args({5000, 1})
    ->Args({5000, 5})
    ->Args({5000, 10})
    ->Args({5000, 40})
    ->Args({5000, 80})
    ->Args({5000, 160})
    ->Args({10000, 1})
    ->Args({10000, 5})
    ->Args({10000, 10})
    ->Args({10000, 40})
    ->Args({20000, 1})
    ->Args({20000, 5})
    ->Args({20000, 10})
    ->Args({20000, 40})
    ->Args({40000, 1})
    ->Args({40000, 5})
    ->Args({40000, 10})
    ->Args({40000, 40});

static void TestPartitionWork(std::ptrdiff_t ThreadId, std::ptrdiff_t ThreadCount, std::ptrdiff_t TotalWork,
                              std::ptrdiff_t* WorkIndex, std::ptrdiff_t* WorkRemaining) {
  const std::ptrdiff_t WorkPerThread = TotalWork / ThreadCount;
  const std::ptrdiff_t WorkPerThreadExtra = TotalWork % ThreadCount;

  if (ThreadId < WorkPerThreadExtra) {
    *WorkIndex = (WorkPerThread + 1) * ThreadId;
    *WorkRemaining = WorkPerThread + 1;
  } else {
    *WorkIndex = WorkPerThread * ThreadId + WorkPerThreadExtra;
    *WorkRemaining = WorkPerThread;
  }
}

static void BM_GeluBatchParallelFor(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));

  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  const int num_batches = 4;
  for (auto _ : state) {
    tp->SimpleParallelFor(num_batches, [&](std::ptrdiff_t batch_index) {
      std::ptrdiff_t start, work_remaining;
      TestPartitionWork(batch_index, num_batches, batch_size, &start, &work_remaining);
      float* output_ptr = output + start;
      onnxruntime::ConstEigenVectorArrayMap<float> xm(data + start, work_remaining);
      onnxruntime::EigenVectorArrayMap<float> ym(output_ptr, work_remaining);
      ym = xm * static_cast<float>(M_SQRT1_2);
      MlasComputeErf(output_ptr, output_ptr, work_remaining);
      ym = xm * 0.5f * (ym + 1.0f);
    });
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_GeluBatchParallelFor)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000);

//The one we're currently using
static void BM_GeluBatchParallelFor2(benchmark::State& state) {
  const size_t elem_count = static_cast<size_t>(state.range(0));

  float* output_data = (float*)aligned_alloc(sizeof(float) * elem_count, 64);
  float* input_data = GenerateArrayWithRandomValue<float>(elem_count, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));

  using T = float;
  static const int64_t length_per_task = 4096;  // this number comes from FastGelu.
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
  for (auto _ : state) {
    concurrency::ThreadPool::TryBatchParallelFor(
        tp.get(), static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min<int64_t>(length_per_task, elem_count - start);

          for (int64_t i = 0; i < count; i++) {
            T value = p_input[i];
            p_output[i] = value * static_cast<T>(M_SQRT1_2);
          }

          MlasComputeErf(p_output, p_output, count);

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
        },
        0);
  }
  aligned_free(input_data);
  aligned_free(output_data);
}

BENCHMARK(BM_GeluBatchParallelFor2)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);

static void BM_GeluBatchParallelFor3(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  concurrency::ThreadPool::SchedulingParams p(concurrency::ThreadPool::SchedulingStrategy::kFixedBlockSize, optional<int64_t>(), 4096);
  for (auto _ : state) {
    tp->ParallelFor(batch_size, p, [data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      float* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<float> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<float> ym(output_ptr, len);
      ym = xm * static_cast<float>(M_SQRT1_2);
      MlasComputeErf(output_ptr, output_ptr, len);
      ym = xm * 0.5f * (ym + 1.0f);
    });
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_GeluBatchParallelFor3)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);