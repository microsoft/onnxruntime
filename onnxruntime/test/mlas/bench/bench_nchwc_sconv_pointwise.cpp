// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/util/thread_utils.h"

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

enum class PointwiseBenchmarkPath {
  NchwConv,
  NchwcBaseline,
  NchwcBaselineInputBatch128,
  NchwcBaselineInputBatch256,
  NchwcBaselineInputBatch128FilterSet2,
  NchwcBaselineInputBatch128FilterSet4,
  NchwcBaselineInputBatch128FilterSet4OutputChunk3,
  NchwcSiluUnfused,
  NchwcSiluFused,
  NchwcGeluUnfused,
  NchwcGeluFused,
};

bool IsFusedPointwiseSiluBenchmarkAvailable() {
#if defined(MLAS_TARGET_AMD64)
  return MlasNchwcGetBlockSize() > 1 &&
         GetMlasPlatform().ConvPointwiseFloatKernel == MlasConvPointwiseFloatKernelAvx512F;
#else
  return false;
#endif
}

bool IsFusedPointwiseGeluBenchmarkAvailable() {
  return IsFusedPointwiseSiluBenchmarkAvailable();
}

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

MLAS_BACKEND_KERNEL_SELECTOR_CONFIG MakeMlasConfig(size_t pointwise_input_channel_batch_override = 0,
                                                  size_t nchwc_filter_set_size_override = 0,
                                                  size_t pointwise_output_count_chunk_override = 0) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.pointwise_input_channel_batch_override = pointwise_input_channel_batch_override;
  config.nchwc_filter_set_size_override = nchwc_filter_set_size_override;
  config.pointwise_output_count_chunk_override = pointwise_output_count_chunk_override;
  return config;
}

void ValidatePointwiseArgs(int64_t batch_count,
                           int64_t input_channels,
                           int64_t input_height,
                           int64_t input_width,
                           int64_t output_channels,
                           int64_t stride) {
  if (batch_count <= 0 || input_channels <= 0 || input_height <= 0 || input_width <= 0 ||
      output_channels <= 0 || stride <= 0) {
    throw std::invalid_argument("All pointwise benchmark dimensions must be positive.");
  }
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

void RunNchwPointwise(benchmark::State& state,
                      MLAS_THREADPOOL* thread_pool) {
  const int64_t batch_count = state.range(0);
  const int64_t input_channels = state.range(1);
  const int64_t input_height = state.range(2);
  const int64_t input_width = state.range(3);
  const int64_t output_channels = state.range(4);
  const int64_t stride = state.range(5);

  ValidatePointwiseArgs(batch_count, input_channels, input_height, input_width, output_channels, stride);

  const int64_t kernel_shape[] = {1, 1};
  const int64_t dilation_shape[] = {1, 1};
  const int64_t padding[] = {0, 0, 0, 0};
  const int64_t stride_shape[] = {stride, stride};

  const int64_t output_height = (input_height - 1) / stride + 1;
  const int64_t output_width = (input_width - 1) / stride + 1;

  const std::vector<int64_t> input_shape = {input_height, input_width};
  const std::vector<int64_t> output_shape = {output_height, output_width};
  const std::vector<int64_t> x_shape = {batch_count, input_channels, input_height, input_width};
  const std::vector<int64_t> f_shape = {output_channels, input_channels, 1, 1};

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  MLAS_CONV_PARAMETERS parameters;
  size_t working_buffer_size = 0;
  MlasConvPrepare(&parameters,
                  2,
                  static_cast<size_t>(batch_count),
                  1,
                  static_cast<size_t>(input_channels),
                  input_shape.data(),
                  kernel_shape,
                  dilation_shape,
                  padding,
                  stride_shape,
                  output_shape.data(),
                  static_cast<size_t>(output_channels),
                  &activation,
                  &working_buffer_size,
                  0.0f,
                  thread_pool);

  auto input_nchw = RandomVectorUniform(x_shape, -1.0f, 1.0f);
  auto filter_oihw = RandomVectorUniform(f_shape, -1.0f, 1.0f);
  std::vector<float> bias(static_cast<size_t>(output_channels), 0.0f);
  std::vector<float> output_nchw(static_cast<size_t>(batch_count * output_channels * output_height * output_width));
  std::vector<float> working_buffer(working_buffer_size);

  MlasConv(&parameters,
           input_nchw.data(),
           filter_oihw.data(),
           bias.data(),
           working_buffer.empty() ? nullptr : working_buffer.data(),
           output_nchw.data(),
           thread_pool);

  for (auto _ : state) {
    MlasConv(&parameters,
             input_nchw.data(),
             filter_oihw.data(),
             bias.data(),
             working_buffer.empty() ? nullptr : working_buffer.data(),
             output_nchw.data(),
             thread_pool);
  }
}

void RunNchwcPointwise(benchmark::State& state,
                       MLAS_THREADPOOL* thread_pool,
                       size_t pointwise_input_channel_batch_override = 0,
                       size_t nchwc_filter_set_size_override = 0,
                       size_t pointwise_output_count_chunk_override = 0) {
  const int64_t batch_count = state.range(0);
  const int64_t input_channels = state.range(1);
  const int64_t input_height = state.range(2);
  const int64_t input_width = state.range(3);
  const int64_t output_channels = state.range(4);
  const int64_t stride = state.range(5);

  ValidatePointwiseArgs(batch_count, input_channels, input_height, input_width, output_channels, stride);

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

  auto mlas_backend_kernel_selector_config = MakeMlasConfig(pointwise_input_channel_batch_override,
                                                            nchwc_filter_set_size_override,
                                                            pointwise_output_count_chunk_override);

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

void RunNchwcPointwiseWithSilu(benchmark::State& state,
                              MLAS_THREADPOOL* thread_pool,
                              bool fused_activation) {
  const int64_t batch_count = state.range(0);
  const int64_t input_channels = state.range(1);
  const int64_t input_height = state.range(2);
  const int64_t input_width = state.range(3);
  const int64_t output_channels = state.range(4);
  const int64_t stride = state.range(5);

  ValidatePointwiseArgs(batch_count, input_channels, input_height, input_width, output_channels, stride);

  const size_t block_size = MlasNchwcGetBlockSize();
  if (block_size <= 1) {
    state.SkipWithError("NCHWc is not supported on this platform.");
    return;
  }

  if ((static_cast<size_t>(input_channels) % 4) != 0) {
    state.SkipWithError("Input channel count must be a multiple of 4 for NCHWc reorder.");
    return;
  }

  if (fused_activation && !IsFusedPointwiseSiluBenchmarkAvailable()) {
    state.SkipWithError("Fused AVX512 pointwise SiLU path is not available on this platform.");
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
  const size_t output_elements = static_cast<size_t>(batch_count) * static_cast<size_t>(output_channels) * output_size;
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
  std::vector<float> activation_output(output_nchwc.size());

  ReorderInputToNchwc(input_nchw.data(),
                      input_nchwc.data(),
                      static_cast<size_t>(batch_count),
                      static_cast<size_t>(input_channels),
                      aligned_input_channels,
                      input_size);
  MlasReorderFilterOIHWBiBo(filter_shape, filter_oihw.data(), filter_nchwc.data());

  MLAS_ACTIVATION activation;
  activation.ActivationKind = fused_activation ? MlasSiluActivation : MlasIdentityActivation;

  auto mlas_backend_kernel_selector_config = MakeMlasConfig();

  auto invoke = [&]() {
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

    if (!fused_activation) {
      MlasComputeSilu(output_nchwc.data(), activation_output.data(), output_elements);
      benchmark::DoNotOptimize(activation_output.data());
    } else {
      benchmark::DoNotOptimize(output_nchwc.data());
    }

    benchmark::ClobberMemory();
  };

  invoke();

  for (auto _ : state) {
    invoke();
  }
}

void RunNchwcPointwiseWithGelu(benchmark::State& state,
                              MLAS_THREADPOOL* thread_pool,
                              bool fused_activation) {
  const int64_t batch_count = state.range(0);
  const int64_t input_channels = state.range(1);
  const int64_t input_height = state.range(2);
  const int64_t input_width = state.range(3);
  const int64_t output_channels = state.range(4);
  const int64_t stride = state.range(5);

  ValidatePointwiseArgs(batch_count, input_channels, input_height, input_width, output_channels, stride);

  const size_t block_size = MlasNchwcGetBlockSize();
  if (block_size <= 1) {
    state.SkipWithError("NCHWc is not supported on this platform.");
    return;
  }

  if ((static_cast<size_t>(input_channels) % 4) != 0) {
    state.SkipWithError("Input channel count must be a multiple of 4 for NCHWc reorder.");
    return;
  }

  if (fused_activation && !IsFusedPointwiseGeluBenchmarkAvailable()) {
    state.SkipWithError("Fused AVX512 pointwise Gelu path is not available on this platform.");
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
  const size_t output_elements = static_cast<size_t>(batch_count) * static_cast<size_t>(output_channels) * output_size;
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
  std::vector<float> activation_output(output_nchwc.size());

  ReorderInputToNchwc(input_nchw.data(),
                      input_nchwc.data(),
                      static_cast<size_t>(batch_count),
                      static_cast<size_t>(input_channels),
                      aligned_input_channels,
                      input_size);
  MlasReorderFilterOIHWBiBo(filter_shape, filter_oihw.data(), filter_nchwc.data());

  MLAS_ACTIVATION activation;
  activation.ActivationKind = fused_activation ? MlasGeluErfActivation : MlasIdentityActivation;

  auto mlas_backend_kernel_selector_config = MakeMlasConfig();

  auto invoke = [&]() {
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

    if (!fused_activation) {
      MlasComputeGeluErf(output_nchwc.data(), activation_output.data(), output_elements);
      benchmark::DoNotOptimize(activation_output.data());
    } else {
      benchmark::DoNotOptimize(output_nchwc.data());
    }

    benchmark::ClobberMemory();
  };

  invoke();

  for (auto _ : state) {
    invoke();
  }
}

void RunPointwise(benchmark::State& state,
                  MLAS_THREADPOOL* thread_pool,
                  PointwiseBenchmarkPath path) {
  switch (path) {
    case PointwiseBenchmarkPath::NchwConv:
      RunNchwPointwise(state, thread_pool);
      break;
    case PointwiseBenchmarkPath::NchwcBaseline:
      RunNchwcPointwise(state, thread_pool);
      break;
    case PointwiseBenchmarkPath::NchwcBaselineInputBatch128:
      RunNchwcPointwise(state, thread_pool, 128);
      break;
    case PointwiseBenchmarkPath::NchwcBaselineInputBatch256:
      RunNchwcPointwise(state, thread_pool, 256);
      break;
    case PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet2:
      RunNchwcPointwise(state, thread_pool, 128, 2);
      break;
    case PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet4:
      RunNchwcPointwise(state, thread_pool, 128, 4);
      break;
    case PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet4OutputChunk3:
      RunNchwcPointwise(state, thread_pool, 128, 4, 3);
      break;
    case PointwiseBenchmarkPath::NchwcSiluUnfused:
      RunNchwcPointwiseWithSilu(state, thread_pool, false);
      break;
    case PointwiseBenchmarkPath::NchwcSiluFused:
      RunNchwcPointwiseWithSilu(state, thread_pool, true);
      break;
    case PointwiseBenchmarkPath::NchwcGeluUnfused:
      RunNchwcPointwiseWithGelu(state, thread_pool, false);
      break;
    case PointwiseBenchmarkPath::NchwcGeluFused:
      RunNchwcPointwiseWithGelu(state, thread_pool, true);
      break;
  }
}

void NCHW_POINTWISE(benchmark::State& state) {
  RunPointwise(state, nullptr, PointwiseBenchmarkPath::NchwConv);
}

void NCHW_POINTWISE_THREADED(benchmark::State& state) {
  RunPointwise(state, GetMlasThreadPoolForNchwcPointwiseBenchmark(), PointwiseBenchmarkPath::NchwConv);
}

void NCHW_POINTWISE_MODEL1(benchmark::State& state) {
  RunPointwise(state, nullptr, PointwiseBenchmarkPath::NchwConv);
}

void NCHW_POINTWISE_THREADED_MODEL1(benchmark::State& state) {
  RunPointwise(state, GetMlasThreadPoolForNchwcPointwiseBenchmark(), PointwiseBenchmarkPath::NchwConv);
}

void NCHWC_POINTWISE(benchmark::State& state, PointwiseBenchmarkPath path) {
  RunPointwise(state,
               nullptr,
               path);
}

void NCHWC_POINTWISE_THREADED(benchmark::State& state, PointwiseBenchmarkPath path) {
  RunPointwise(state,
               GetMlasThreadPoolForNchwcPointwiseBenchmark(),
               path);
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

void Model1Pointwise(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(ArgNamesForNchwcPointwise());

  benchmark->Args({1, 64, 64, 64, 192, 1});
  benchmark->Args({1, 192, 64, 64, 64, 1});
  benchmark->Args({1, 128, 32, 32, 384, 1});
  benchmark->Args({1, 384, 32, 32, 128, 1});
  benchmark->Args({1, 256, 16, 16, 768, 1});
  benchmark->Args({1, 768, 16, 16, 256, 1});
  benchmark->Args({1, 512, 8, 8, 1536, 1});
  benchmark->Args({1, 1536, 8, 8, 512, 1});
  benchmark->Args({1, 512, 8, 8, 512, 1});
}

void Model1ProjectionPointwise(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(ArgNamesForNchwcPointwise());

  benchmark->Args({1, 192, 64, 64, 64, 1});
  benchmark->Args({1, 384, 32, 32, 128, 1});
  benchmark->Args({1, 768, 16, 16, 256, 1});
  benchmark->Args({1, 1536, 8, 8, 512, 1});
}

void Model2QuickGeluRegressionPointwise(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(ArgNamesForNchwcPointwise());
  benchmark->Args({1, 48, 104, 104, 32, 1});
  benchmark->Args({1, 48, 104, 104, 48, 1});
  benchmark->Args({1, 96, 52, 52, 48, 1});
  benchmark->Args({1, 96, 52, 52, 96, 1});
  benchmark->Args({1, 192, 26, 26, 96, 1});
  benchmark->Args({1, 192, 26, 26, 192, 1});
  benchmark->Args({1, 384, 13, 13, 192, 1});
  benchmark->Args({1, 384, 13, 13, 384, 1});
  benchmark->Args({1, 768, 13, 13, 384, 1});
}

void Model1GeluPointwise(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(ArgNamesForNchwcPointwise());
  benchmark->Args({1, 64, 64, 64, 192, 1});
  benchmark->Args({1, 192, 64, 64, 64, 1});
  benchmark->Args({1, 128, 32, 32, 384, 1});
  benchmark->Args({1, 384, 32, 32, 128, 1});
  benchmark->Args({1, 256, 16, 16, 768, 1});
  benchmark->Args({1, 768, 16, 16, 256, 1});
  benchmark->Args({1, 512, 8, 8, 1536, 1});
  benchmark->Args({1, 1536, 8, 8, 512, 1});
  benchmark->Args({1, 512, 8, 8, 512, 1});
}

}  // namespace

BENCHMARK(NCHW_POINTWISE)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE)->Apply(Model1Pointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, ResNetLikeBaseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, HighChannelsBaseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE_MODEL1)->Apply(Model1Pointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1Baseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(Model1Pointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1ProjectionBaseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1ProjectionInputBatch128, PointwiseBenchmarkPath::NchwcBaselineInputBatch128)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1ProjectionInputBatch256, PointwiseBenchmarkPath::NchwcBaselineInputBatch256)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1ProjectionInputBatch128FilterSet2, PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet2)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1ProjectionInputBatch128FilterSet4, PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet4)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1ProjectionInputBatch128FilterSet4OutputChunk3, PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet4OutputChunk3)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE_THREADED)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE_THREADED)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE_THREADED)->Apply(Model1Pointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE_THREADED)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, ResNetLikeBaseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(ResNetPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, HighChannelsBaseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(HighChannelPointwise)->UseRealTime();
BENCHMARK(NCHW_POINTWISE_THREADED_MODEL1)->Apply(Model1Pointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1Baseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(Model1Pointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1ProjectionBaseline, PointwiseBenchmarkPath::NchwcBaseline)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1ProjectionInputBatch128, PointwiseBenchmarkPath::NchwcBaselineInputBatch128)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1ProjectionInputBatch256, PointwiseBenchmarkPath::NchwcBaselineInputBatch256)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1ProjectionInputBatch128FilterSet2, PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet2)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1ProjectionInputBatch128FilterSet4, PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet4)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1ProjectionInputBatch128FilterSet4OutputChunk3, PointwiseBenchmarkPath::NchwcBaselineInputBatch128FilterSet4OutputChunk3)->Apply(Model1ProjectionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model2SiluUnfused, PointwiseBenchmarkPath::NchwcSiluUnfused)->Apply(Model2QuickGeluRegressionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model2SiluFused, PointwiseBenchmarkPath::NchwcSiluFused)->Apply(Model2QuickGeluRegressionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model2SiluUnfused, PointwiseBenchmarkPath::NchwcSiluUnfused)->Apply(Model2QuickGeluRegressionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model2SiluFused, PointwiseBenchmarkPath::NchwcSiluFused)->Apply(Model2QuickGeluRegressionPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1GeluUnfused, PointwiseBenchmarkPath::NchwcGeluUnfused)->Apply(Model1GeluPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE, Model1GeluFused, PointwiseBenchmarkPath::NchwcGeluFused)->Apply(Model1GeluPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1GeluUnfused, PointwiseBenchmarkPath::NchwcGeluUnfused)->Apply(Model1GeluPointwise)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_POINTWISE_THREADED, Model1GeluFused, PointwiseBenchmarkPath::NchwcGeluFused)->Apply(Model1GeluPointwise)->UseRealTime();
