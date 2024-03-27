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
  const auto Aligned = narrow<bool>(state.range(0));
  const auto N = narrow<int>(state.range(1));
  const auto D = narrow<int>(state.range(2));
  const auto Threads = narrow<int>(state.range(3));

  if (N <= 0 || D <= 0 || Threads <= 0) {
    throw std::invalid_argument("N, D, and Threads must be greater than 0!");
  }

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = Threads;
  tpo.auto_set_affinity = true;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto buffer = RandomVectorUniform<float>(static_cast<size_t>(N * D + 32 + 1), -1.0f, 1.0f);

  const float* input = nullptr;
  float* output = nullptr;
  if (Aligned) {
    input = reinterpret_cast<const float*>((reinterpret_cast<uintptr_t>(buffer.data()) + 32) & ~31);
    output = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(buffer.data()) + 32) & ~31);
  } else {
    input = reinterpret_cast<const float*>(((reinterpret_cast<uintptr_t>(buffer.data()) + 32) & ~31) + 1);
    output = reinterpret_cast<float*>(((reinterpret_cast<uintptr_t>(buffer.data()) + 32) & ~31) + 1);
  }

  // warm up run
  MlasComputeSoftmax(input, output, N, D, false, tp.get());

  for (auto _ : state) {
    MlasComputeSoftmax(input, output, N, D, false, tp.get());
  }
}

static void ComputeSoftmaxInplaceArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"Aligned", "N", "D", "Threads"});

  b->ArgsProduct({
      {true, false},  // aligned
      {240000},       // N
      {31, 32}, //{15, 2000},   // D
      {1, 8},         // Threads
  });
}

BENCHMARK(COMPUTESOFTMAXINPLACE)->Apply(ComputeSoftmaxInplaceArgs)->UseRealTime();
