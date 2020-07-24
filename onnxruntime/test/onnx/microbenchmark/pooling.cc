// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/graph/onnx_protobuf.h>
#include <benchmark/benchmark.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/util/thread_utils.h>
#include <core/providers/cpu/nn/pool_functors.h>

#include <mlas.h>
#include <random>

extern OrtEnv* env;
extern const OrtApi* g_ort;
using namespace onnxruntime::concurrency;

static int64_t CalcSize(int64_t* shape, size_t len) {
  int64_t ret = 1;
  for (size_t i = 0; i != len; ++i) ret *= shape[i];
  return ret;
}

static void RunMlasPool2D(const OrtThreadPoolParams& param, int64_t batch_size, benchmark::State& state) {
  std::unique_ptr<ThreadPool> tp = CreateThreadPool(&onnxruntime::Env::Default(), param, onnxruntime::concurrency::ThreadPoolType::INTRA_OP);
  int64_t input_shape[] = {1, 64, 112, 112};
  int64_t kernel_shape[] = {3, 3};
  int64_t padding[] = {0, 0, 1, 1};
  int64_t stride_shape[] = {2, 2};
  int64_t output_shape[] = {1, 64, 56, 56};
  input_shape[0] = batch_size;
  output_shape[0] = batch_size;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-10, 10);
  std::vector<float> input(CalcSize(input_shape, 4));
  std::vector<float> output(CalcSize(output_shape, 4));

  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = dist(gen);
  }
  for (size_t i = 0; i != output.size(); ++i) {
    output[i] = dist(gen);
  }
  for (auto _ : state) {
    MlasPool(MlasMaximumPooling, 2, input_shape, kernel_shape, padding, stride_shape, output_shape, input.data(),
             output.data(), tp.get());
  }
}

static void BM_MlasPoolWithSpinAndAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = true;
  param.allow_spinning = true;
  RunMlasPool2D(param, batch_size, state);
}

static void BM_MlasPoolWithSpinNoAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = false;
  param.allow_spinning = true;
  RunMlasPool2D(param, batch_size, state);
}

static void BM_MlasPoolNoSpinNoAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = false;
  param.allow_spinning = false;
  RunMlasPool2D(param, batch_size, state);
}

static void BM_MlasPoolNoSpinWithAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = true;
  param.allow_spinning = false;
  RunMlasPool2D(param, batch_size, state);
}

BENCHMARK(BM_MlasPoolWithSpinAndAffinity)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(4)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_MlasPoolWithSpinNoAffinity)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(4)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_MlasPoolNoSpinWithAffinity)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(4)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_MlasPoolNoSpinNoAffinity)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(4)
    ->Unit(benchmark::TimeUnit::kMicrosecond);

static void RunPool2D(const OrtThreadPoolParams& param, int64_t batch_size, benchmark::State& state) {
  std::unique_ptr<ThreadPool> tp = CreateThreadPool(&onnxruntime::Env::Default(), param, onnxruntime::concurrency::ThreadPoolType::INTRA_OP);
  int64_t input_shape[] = {1, 64, 112, 112};
  std::vector<int64_t> kernel_shape = {3, 3};
  std::vector<int64_t> padding = {0, 0, 1, 1};
  // int64_t stride_shape[] = { 2, 2 };
  int64_t output_shape[] = {1, 64, 56, 56};
  input_shape[0] = batch_size;
  output_shape[0] = batch_size;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-10, 10);
  std::vector<float> input(CalcSize(input_shape, 4));
  std::vector<float> output(CalcSize(output_shape, 4));

  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = dist(gen);
  }
  for (size_t i = 0; i != output.size(); ++i) {
    output[i] = dist(gen);
  }
  int64_t x_step = input_shape[2] * input_shape[3];
  //TODO: no hard code
  int64_t y_step = 3136;

  onnxruntime::MaxPool2DTask<float> task{input.data(), output.data(), nullptr, x_step, y_step, 1, 1, 56, 56, 2, 2, 112, 112, kernel_shape, padding, 0};

  for (auto _ : state)
    ThreadPool::TryParallelFor(tp.get(), input_shape[0] * input_shape[1], task.Cost(), task);
}
static void BM_Pool2D(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = true;
  param.allow_spinning = true;
  RunPool2D(param, batch_size, state);
}

static void BM_PoolSingleThread(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.thread_pool_size = 1;
  RunPool2D(param, batch_size, state);
}

BENCHMARK(BM_Pool2D)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(4)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_PoolSingleThread)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(4)
    ->Unit(benchmark::TimeUnit::kMicrosecond);