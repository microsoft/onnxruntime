// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Compares the assembly and sliding window AVX-512 NCHWc depthwise convolution kernels
// (MLAS_BACKEND_KERNEL_SELECTOR_CONFIG::nchwc_depthwise_sliding_kernel).

#include "mlas.h"
#include "bench_util.h"
#include "core/platform/threadpool.h"
#include "core/util/thread_utils.h"

#include <map>
#include <memory>
#include <vector>

namespace {

MLAS_THREADPOOL* GetDepthwiseThreadPool(size_t threads) {
  if (threads <= 1) {
    return nullptr;
  }
  static std::map<size_t, std::unique_ptr<onnxruntime::concurrency::ThreadPool>> pools;
  auto& pool = pools[threads];
  if (!pool) {
    pool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
        &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, static_cast<int>(threads), true);
  }
  return pool.get();
}

// Args: channels, spatial side, kernel size, threads, sliding (0/1).
void NCHWC_DEPTHWISE(benchmark::State& state) {
  // Without the sliding window kernel both settings evaluate the assembly kernel, which would
  // report a meaningless comparison.
  if (!MlasNchwcDepthwiseSlidingKernelAvailable()) {
    state.SkipWithError("The sliding window NCHWc depthwise kernel is not available on this platform.");
    return;
  }

  const int64_t c = state.range(0);
  const int64_t s = state.range(1);
  const int64_t k = state.range(2);
  MLAS_THREADPOOL* tp = GetDepthwiseThreadPool(static_cast<size_t>(state.range(3)));
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG cfg;
  cfg.nchwc_depthwise_sliding_kernel = state.range(4) != 0;

  const std::vector<float> input = RandomVectorUniform(size_t(c * s * s), -1.0f, 1.0f);
  const std::vector<float> filter = RandomVectorUniform(size_t(c * k * k), -0.5f, 0.5f);
  const std::vector<float> bias = RandomVectorUniform(size_t(c), -0.1f, 0.1f);
  std::vector<float> output(input.size());

  const int64_t shape[] = {1, c, s, s};
  const int64_t kernel[] = {k, k};
  const int64_t one[] = {1, 1};
  const int64_t pads[] = {k / 2, k / 2, k / 2, k / 2};
  MLAS_ACTIVATION identity;
  identity.ActivationKind = MlasIdentityActivation;

  auto run = [&]() {
    MlasNchwcConv(shape, kernel, one, pads, one, shape, size_t(c), input.data(), filter.data(), bias.data(),
                  output.data(), &identity, true, tp, &cfg, false);
  };

  run();  // warm up
  for (auto _ : state) {
    run();
  }
}

void DepthwiseArgs(benchmark::Benchmark* b) {
  b->ArgNames({"C", "HW", "K", "Threads", "Sliding"});
  const std::vector<std::pair<int64_t, int64_t>> shapes = {
      {64, 64},
      {128, 32},
      {256, 16},
      {512, 8},
      {32, 112},
      {96, 56},
      {240, 28},
  };
  for (const auto& shape : shapes) {
    for (int64_t k : {3, 5, 7}) {
      for (int64_t threads : {1, 4}) {
        for (int64_t sliding : {0, 1}) {
          b->Args({shape.first, shape.second, k, threads, sliding});
        }
      }
    }
  }
}

}  // namespace

BENCHMARK(NCHWC_DEPTHWISE)->Apply(DepthwiseArgs)->UseRealTime();
