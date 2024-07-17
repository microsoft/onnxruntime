// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <numeric>

#include "core/mlas/inc/mlas_q4.h"
#include "test/mlas/bench/bench_util.h"
#include "core/util/thread_utils.h"

static void BM_QDQBlockwiseQuantizer_QuantizeColumnwise(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);
  int quant_block_size = state.range(2);
  int threads = state.range(3);
  size_t scale_size = (M + quant_block_size - 1) / quant_block_size * N;

  auto src = RandomVectorUniform(M * N, -16.0f, 14.0f);
  auto scales = std::vector<float>(scale_size);
  auto zero_points = std::vector<uint8_t>((scale_size + 1) / 2);
  auto dst = std::vector<uint8_t>((M * N + 1) / 2);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    benchmark::DoNotOptimize(dst.data());
    MlasQDQQuantizeBlockwise<float, 4>(
        src.data(), scales.data(), zero_points.data(), dst.data(),
        true, M, N, quant_block_size, tp.get());
    benchmark::ClobberMemory();
  }
}

static void BM_MlasQuantizeBlockwise(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);
  int quant_block_size = state.range(2);
  int threads = state.range(3);
  size_t scale_size = (M + quant_block_size - 1) / quant_block_size * N;

  auto src = RandomVectorUniform(M * N, -16.0f, 14.0f);
  auto scales = std::vector<float>(scale_size);
  auto zero_points = std::vector<uint8_t>((scale_size + 1) / 2);
  auto dst = std::vector<uint8_t>((M * N + 1) / 2);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    benchmark::DoNotOptimize(dst.data());
    MlasQuantizeBlockwise<float, 4>(
        dst.data(), scales.data(), zero_points.data(), src.data(),
        quant_block_size, true, M, N, N, tp.get());
    benchmark::ClobberMemory();
  }
}

static void BM_QDQBlockwiseQuantizer_TransposeColumnwise(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);
  int quant_block_size = state.range(2);
  int threads = state.range(3);
  bool add8 = state.range(4) != 0;
  int quant_num_M = (M + quant_block_size - 1) / quant_block_size;
  int blob_size = (quant_block_size + 1) / 2;
  size_t scale_size = quant_num_M * N;

  auto scales = RandomVectorUniform<float>(scale_size, -16.0f, 14.0f);
  auto zero_points = RandomVectorUniform<uint8_t>(static_cast<size_t>((scale_size + 1) / 2), 0, 255);
  auto dst = RandomVectorUniform<uint8_t>(static_cast<size_t>((M * N + 1) / 2), 0, 255);
  auto scales_T = std::vector<float>(scale_size);
  auto zero_points_T = std::vector<uint8_t>(((quant_num_M + 1) / 2) * N);
  auto dst_T = std::vector<uint8_t>(quant_num_M * blob_size * N);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  if (add8) {
    for (auto _ : state) {
      benchmark::DoNotOptimize(dst.data());
      MlasQDQTransposeBlockwiseQuantized<float, 4, true>(
          dst.data(), scales.data(), zero_points.data(), dst_T.data(), scales_T.data(), zero_points_T.data(),
          true, M, N, quant_block_size, tp.get());
      benchmark::ClobberMemory();
    }
  } else {
    for (auto _ : state) {
      benchmark::DoNotOptimize(dst.data());
      MlasQDQTransposeBlockwiseQuantized<float, 4, false>(
          dst.data(), scales.data(), zero_points.data(), dst_T.data(), scales_T.data(), zero_points_T.data(),
          true, M, N, quant_block_size, tp.get());
      benchmark::ClobberMemory();
    }
  }
}

BENCHMARK(BM_QDQBlockwiseQuantizer_QuantizeColumnwise)
    ->UseRealTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"M", "N", "quant_block_size", "threads"});
      b->ArgsProduct({{1024, 4096}, {4096, 4095}, {64, 128}, {8}});
    });

BENCHMARK(BM_MlasQuantizeBlockwise)
    ->UseRealTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"M", "N", "quant_block_size", "threads"});
      b->ArgsProduct({{1024, 4096}, {4096, 4095}, {64, 128}, {8}});
    });

BENCHMARK(BM_QDQBlockwiseQuantizer_TransposeColumnwise)
    ->UseRealTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"M", "N", "quant_block_size", "threads", "add8"});
      b->ArgsProduct({{1024, 4096}, {4096, 4095}, {64, 128}, {2, 8, 16}, {0, 1}});
    });
