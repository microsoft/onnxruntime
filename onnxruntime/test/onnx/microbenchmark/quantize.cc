#include "common.h"

#include <benchmark/benchmark.h>
#include "core/util/qmath.h"
#include "core/util/thread_utils.h"
#include "core/framework/int4.h"

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
    onnxruntime::ParQuantizeLinearStd(a_data, a_data_quant, batch_size, scale, zero, tp.get());
    benchmark::ClobberMemory();
  }
  aligned_free(a_data_quant);
  aligned_free(a_data);
}

BENCHMARK(BM_Quantize)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Apply(BenchSize);

static void BM_BlockedQuantize_NotLastAxis(benchmark::State& state) {
  using Int4 = onnxruntime::Int4x2;
  using UnpackedType = Int4::UnpackedType;
  const int64_t M = state.range(0);
  const int64_t N = state.range(1);
  const int64_t block_size = state.range(2);
  const int64_t threads = state.range(3);
  size_t batch_size = M * N;
  size_t quant_block_size = 64;
  size_t scale_size = batch_size / quant_block_size;

  float* a_data = GenerateArrayWithRandomValue<float>(batch_size, -16, 14);
  size_t a_quant_size = sizeof(Int4::UnpackedType) * Int4::CalcNumInt4Pairs(batch_size);
  float* scale = GenerateArrayWithRandomValue<float>(scale_size, 1.95f, 2.33f);
  UnpackedType* zero_point = GenerateArrayWithRandomValue<UnpackedType>(Int4::CalcNumInt4Pairs(scale_size), -1, 1);
  UnpackedType* a_data_quant = static_cast<UnpackedType*>(aligned_alloc(a_quant_size, 64));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    benchmark::DoNotOptimize(a_data_quant);
    onnxruntime::BlockedQuantizeLinear<float, Int4, 2>::opNotLastAxis(
        tp.get(), a_data, scale, reinterpret_cast<Int4*>(zero_point), reinterpret_cast<Int4*>(a_data_quant),
        1, M, N, static_cast<std::ptrdiff_t>(quant_block_size),
        static_cast<std::ptrdiff_t>(block_size), true);
    benchmark::ClobberMemory();
  }
  aligned_free(a_data_quant);
  aligned_free(a_data);
  aligned_free(scale);
  aligned_free(zero_point);
}

static void BM_BlockedQuantize_LastAxis(benchmark::State& state) {
  using Int4 = onnxruntime::Int4x2;
  using UnpackedType = Int4::UnpackedType;
  const int64_t M = state.range(0);
  const int64_t N = state.range(1);
  const int64_t quant_block_size = state.range(2);
  const int64_t threads = state.range(3);
  size_t batch_size = M * N;
  size_t scale_size = batch_size / quant_block_size;

  float* a_data = GenerateArrayWithRandomValue<float>(batch_size, -16, 14);
  size_t a_quant_size = sizeof(Int4::UnpackedType) * Int4::CalcNumInt4Pairs(batch_size);
  float* scale = GenerateArrayWithRandomValue<float>(scale_size, 1.95f, 2.33f);
  UnpackedType* zero_point = GenerateArrayWithRandomValue<UnpackedType>(Int4::CalcNumInt4Pairs(scale_size), -1, 1);
  UnpackedType* a_data_quant = static_cast<UnpackedType*>(aligned_alloc(a_quant_size, 64));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    benchmark::DoNotOptimize(a_data_quant);
    onnxruntime::BlockedQuantizeLinear<float, Int4, 2>::opLastAxis(
        tp.get(), a_data, scale, reinterpret_cast<Int4*>(zero_point), reinterpret_cast<Int4*>(a_data_quant),
        M, N, static_cast<std::ptrdiff_t>(quant_block_size), true);
    benchmark::ClobberMemory();
  }
  aligned_free(a_data_quant);
  aligned_free(a_data);
  aligned_free(scale);
  aligned_free(zero_point);
}

BENCHMARK(BM_BlockedQuantize_NotLastAxis)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"M", "N", "block_size", "threads"});
      b->ArgsProduct({{1024, 4096}, {4096}, {128}, {2, 8}});
    });

BENCHMARK(BM_BlockedQuantize_LastAxis)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"M", "N", "quant_block_size", "threads"});
      b->ArgsProduct({{1024, 4096}, {4096}, {64, 128}, {2, 8}});
    });
