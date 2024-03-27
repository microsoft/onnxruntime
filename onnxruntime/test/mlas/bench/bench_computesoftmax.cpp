// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "benchmark/benchmark.h"

#include "bench_util.h"
#include "core/common/narrow.h"
#include "core/util/thread_utils.h"
#include "core/platform/env_var_utils.h"

using onnxruntime::narrow;

void COMPUTESOFTMAXINPLACE(benchmark::State& state) {
  if (state.range(0) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("D must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const auto N = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));
  const auto Threads = narrow<int>(state.range(2));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = Threads;
  tpo.auto_set_affinity = true;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto buffer = RandomVectorUniform(static_cast<size_t>(N * D), -1.0f, 1.0f);

  // warm up run
  MlasComputeSoftmax(buffer.data(), buffer.data(), N, D, false, tp.get());

  for (auto _ : state) {
    MlasComputeSoftmax(buffer.data(), buffer.data(), N, D, false, tp.get());
  }
}

static void ComputeSoftmaxInplaceArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "D", "Threads"});

  b->ArgsProduct({
      {240000},     // N
      {15, 2000},   // D
      {8},          // Threads
  });
}

BENCHMARK(COMPUTESOFTMAXINPLACE)->Apply(ComputeSoftmaxInplaceArgs)->UseRealTime();