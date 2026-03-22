// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::string> BuildArgNamesForNchwcPointwise() {
  return {"N", "Cin", "H", "W", "Cout", "Stride"};
}

const std::vector<std::string>& ArgNamesForNchwcPointwise() {
  static const std::vector<std::string> arg_names = BuildArgNamesForNchwcPointwise();
  return arg_names;
}

MLAS_THREADPOOL* GetMlasThreadPoolForNchwcPointwiseBenchmark() {
  static auto threadpool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 4, true);
  return threadpool.get();
}

MLAS_BACKEND_KERNEL_SELECTOR_CONFIG MakeMlasConfig(bool enable_nchwc_pointwise_filter_repacking) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.use_nchwc_pointwise_filter_repacking = enable_nchwc_pointwise_filter_repacking;
  return config;
}

void ReorderInputToNchwc(const float* source,
                         float* destination,
                         size_t batch_count,
                         size_t input_channels,
                         size_t aligned_input_channels,
                         size_t input_size) {
  for (size_t batch = 0; batch < batch_count; ++batch) {
    MlasReorderInputNchw(source + batch * input_channels * input_size,
                         destination + batch * aligned_input_channels * input_size,
                         input_channels,
                         input_size);
  }
}

void RunNchwcPointwise(benchmark::State& state,
                       MLAS_THREADPOOL* thread_pool,
                       bool enable_nchwc_pointwise_filter_repacking) {
  const int64_t batch_count = state.range(0);
  const int64_t input_channels = state.range(1);
  const int64_t input_height = state.range(2);
  const int64_t input_width = state.range(3);
  const int64_t output_channels = state.range(4);
  const int64_t stride = state.range(5);

  if (batch_count <= 0 || input_channels <= 0 || input_height <= 0 || input_width <= 0 ||
      output_channels <= 0 || stride <= 0) {
    throw std::invalid_argument("All NCHWc pointwise benchmark dimensions must be positive.");
  }

  const size_t block_size = MlasNchwcGetBlockSize();
  if (block_size <= 1) {
    state.SkipWithError("NCHWc is not supported on this platform.");
    return;
  }

  if ((static_cast<size_t>(input_channels) % 4) != 0) {
    state.SkipWithError("Input channel count must be a multiple of 4 for NCHWc reorder.");
    return;
  }

  const int64_t kernel_shape[] = {1, 1};
  const int64_t dilation_shape[] = {1, 1};
  const int64_t padding[] = {0, 0, 0, 0};
  const int64_t stride_shape[] = {stride, stride};

  const int64_t output_height = (input_height - 1) / stride + 1;
  const int64_t output_width = (input_width - 1) / stride + 1;

  const size_t input_size = static_cast<size_t>(input_height * input_width);
  const size_t output_size = static_cast<size_t>(output_height * output_width);
  const size_t aligned_input_channels = (static_cast<size_t>(input_channels) + block_size - 1) & ~(block_size - 1);
  const size_t aligned_output_channels = (static_cast<size_t>(output_channels) + block_size - 1) & ~(block_size - 1);

  const int64_t input_shape[] = {batch_count, static_cast<int64_t>(aligned_input_channels), input_height, input_width};
  const int64_t filter_shape[] = {output_channels, input_channels, 1, 1};
  const int64_t output_shape[] = {batch_count, static_cast<int64_t>(aligned_output_channels), output_height, output_width};

  std::vector<float> input_nchw = RandomVectorUniform(
      static_cast<size_t>(batch_count * input_channels * input_size), -1.0f, 1.0f);
  std::vector<float> filter_oihw = RandomVectorUniform(
      static_cast<size_t>(output_channels * input_channels), -1.0f, 1.0f);
  std::vector<float> bias(static_cast<size_t>(aligned_output_channels), 0.0f);

  std::vector<float> input_nchwc(static_cast<size_t>(batch_count) * aligned_input_channels * input_size);
  std::vector<float> filter_nchwc(aligned_output_channels * aligned_input_channels);
  std::vector<float> output_nchwc(static_cast<size_t>(batch_count) * aligned_output_channels * output_size);

  ReorderInputToNchwc(input_nchw.data(),
                      input_nchwc.data(),
                      static_cast<size_t>(batch_count),
                      static_cast<size_t>(input_channels),
                      aligned_input_channels,
                      input_size);
  MlasReorderFilterOIHWBiBo(filter_shape, filter_oihw.data(), filter_nchwc.data());

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  auto mlas_backend_kernel_selector_config =
      MakeMlasConfig(enable_nchwc_pointwise_filter_repacking);

  MlasNchwcConv(input_shape,
                kernel_shape,
                dilation_shape,
                padding,
                stride_shape,
                output_shape,
                1,
                input_nchwc.data(),
                filter_nchwc.data(),
                bias.data(),
                output_nchwc.data(),
                &activation,
                true,
                thread_pool,
                &mlas_backend_kernel_selector_config,
                false);

  for (auto _ : state) {
    MlasNchwcConv(input_shape,
                  kernel_shape,
                  dilation_shape,
                  padding,
                  stride_shape,
                  output_shape,
                  1,
                  input_nchwc.data(),
                  filter_nchwc.data(),
                  bias.data(),
                  output_nchwc.data(),
                  &activation,
                  true,
                  thread_pool,
                  &mlas_backend_kernel_selector_config,
                  false);
  }
}

void NCHWC_POINTWISE(benchmark::State& state, bool enable_nchwc_pointwise_filter_repacking) {
  RunNchwcPointwise(state, nullptr, enable_nchwc_pointwise_filter_repacking);
}

void NCHWC_POINTWISE_THREADED(benchmark::State& state, bool enable_nchwc_pointwise_filter_repacking) {
  RunNchwcPointwise(state,
                    GetMlasThreadPoolForNchwcPointwiseBenchmark(),
                    enable_nchwc_pointwise_filter_repacking);
}

void ResNetPointwise(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(ArgNamesForNchwcPointwise());

  benchmark->Args({1, 64, 56, 56, 64, 1});
  benchmark->Args({1, 64, 56, 56, 256, 1});
  benchmark->Args({1, 256, 56, 56, 64, 1});
  benchmark->Args({1, 256, 56, 56, 512, 2});
  benchmark->Args({1, 512, 28, 28, 128, 1});
  benchmark->Args({1, 512, 28, 28, 1024, 2});
  benchmark->Args({1, 1024, 14, 14, 256, 1});
  benchmark->Args({1, 1024, 14, 14, 2048, 2});
  benchmark->Args({1, 2048, 7, 7, 512, 1});
}

void HighChannelPointwise(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(ArgNamesForNchwcPointwise());

  benchmark->Args({1, 256, 28, 28, 256, 1});
  benchmark->Args({1, 256, 28, 28, 512, 1});
  benchmark->Args({1, 512, 14, 14, 512, 1});
  benchmark->Args({1, 512, 14, 14, 1024, 1});
  benchmark->Args({1, 1024, 14, 14, 1024, 1});
  benchmark->Args({1, 1024, 14, 14, 2048, 1});
  benchmark->Args({1, 2048, 7, 7, 2048, 1});
  benchmark->Args({4, 256, 28, 28, 512, 1});
  benchmark->Args({4, 512, 14, 14, 1024, 1});
}

}  // namespace

BENCHMARK_CAPTURE(NCHWC_POINTWISE, ResNetLikeBaseline, false)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, ResNetLikePanelRepack, true)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, HighChannelsBaseline, false)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, HighChannelsPanelRepack, true)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, ResNetLikeBaseline, false)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, ResNetLikePanelRepack, true)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, HighChannelsBaseline, false)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, HighChannelsPanelRepack, true)->Apply(HighChannelPointwise)->UseRealTime();
