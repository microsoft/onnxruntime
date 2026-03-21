// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bench_util.h"

#if defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_ARM_NEON_NCHWC)

#include "../../../core/mlas/lib/mlasi.h"
#include "../../../core/mlas/lib/sconv_nchwc_kernel_neon.h"

#include <cstdint>
#include <vector>

namespace {

constexpr size_t BlockSize = 16;
constexpr unsigned BenchmarkKernelFlags = MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION;

enum class NchwcKernelBenchPath {
  DirectCpp,
  WrapperNoPad,
  WrapperPadded,
#if !defined(_WIN32)
  DirectAsmNoPad,
#endif
};

struct NchwcKernelBenchConfig {
  size_t filter_count;
  size_t output_count_left_pad;
  size_t output_count;
  size_t output_count_right_pad;
  size_t kernel_height;
  size_t kernel_width;
  unsigned kernel_flags;
};

struct NchwcKernelBenchBuffers {
  explicit NchwcKernelBenchBuffers(const NchwcKernelBenchConfig& config)
      : total_output_count(config.output_count_left_pad + config.output_count + config.output_count_right_pad),
        input_width(config.output_count + config.kernel_width - 1),
        total_input_width(config.output_count_left_pad + input_width + config.output_count_right_pad),
        filter_elements_per_block(config.kernel_height * config.kernel_width * BlockSize * BlockSize),
        output_stride_elements(total_output_count * BlockSize),
        stride_width_bytes(BlockSize * sizeof(float)),
        dilation_width_bytes(BlockSize * sizeof(float)),
        input_width_bytes(input_width * BlockSize * sizeof(float)),
        dilated_input_width_bytes(total_input_width * BlockSize * sizeof(float)),
        input_stride_bytes(dilated_input_width_bytes - config.kernel_width * dilation_width_bytes),
        filter_stride_bytes(filter_elements_per_block * sizeof(float)),
        output_stride_bytes(output_stride_elements * sizeof(float)),
        input_storage(RandomVectorUniform(config.kernel_height * total_input_width * BlockSize, -1.0f, 1.0f)),
        filter(RandomVectorUniform(config.filter_count * filter_elements_per_block, -1.0f, 1.0f)),
        bias(RandomVectorUniform(config.filter_count * BlockSize, -0.5f, 0.5f)),
        output(RandomVectorUniform(config.filter_count * output_stride_elements, -0.25f, 0.25f)),
        input_base(input_storage.data() + config.output_count_left_pad * BlockSize),
        input(input_base - config.output_count_left_pad * BlockSize) {}

  const size_t total_output_count;
  const size_t input_width;
  const size_t total_input_width;
  const size_t filter_elements_per_block;
  const size_t output_stride_elements;
  const size_t stride_width_bytes;
  const size_t dilation_width_bytes;
  const size_t input_width_bytes;
  const size_t dilated_input_width_bytes;
  const size_t input_stride_bytes;
  const size_t filter_stride_bytes;
  const size_t output_stride_bytes;
  std::vector<float> input_storage;
  std::vector<float> filter;
  std::vector<float> bias;
  std::vector<float> output;
  const float* input_base;
  const float* input;
};

void RunNchwcKernelBench(
    NchwcKernelBenchPath path,
    const NchwcKernelBenchConfig& config,
    NchwcKernelBenchBuffers& buffers) {
  switch (path) {
    case NchwcKernelBenchPath::DirectCpp:
      // Args: Input, Filter, Output, StrideWidthBytes, DilationWidthBytes,
      // FilterCount, InputStrideBytes, FilterStrideBytes, OutputStrideBytes,
      // KernelHeight, KernelWidth, InputBase, InputWidthBytes,
      // DilatedInputWidthBytes, OutputCountLeftPad, OutputCount,
      // OutputCountRightPad, Bias, KernelFlags.
      MlasConvNchwcFloatKernelNeonCpp(
          buffers.input,
          buffers.filter.data(),
          buffers.output.data(),
          buffers.stride_width_bytes,
          buffers.dilation_width_bytes,
          config.filter_count,
          buffers.input_stride_bytes,
          buffers.filter_stride_bytes,
          buffers.output_stride_bytes,
          config.kernel_height,
          config.kernel_width,
          buffers.input_base,
          buffers.input_width_bytes,
          buffers.dilated_input_width_bytes,
          config.output_count_left_pad,
          config.output_count,
          config.output_count_right_pad,
          buffers.bias.data(),
          config.kernel_flags);
      break;

    case NchwcKernelBenchPath::WrapperNoPad:
    case NchwcKernelBenchPath::WrapperPadded:
      // Same argument order as the direct C++ kernel above. The wrapper uses
      // C++ edges and may route interior spans to asm when supported.
      MlasConvNchwcFloatKernelNeon(
          buffers.input,
          buffers.filter.data(),
          buffers.output.data(),
          buffers.stride_width_bytes,
          buffers.dilation_width_bytes,
          config.filter_count,
          buffers.input_stride_bytes,
          buffers.filter_stride_bytes,
          buffers.output_stride_bytes,
          config.kernel_height,
          config.kernel_width,
          buffers.input_base,
          buffers.input_width_bytes,
          buffers.dilated_input_width_bytes,
          config.output_count_left_pad,
          config.output_count,
          config.output_count_right_pad,
          buffers.bias.data(),
          config.kernel_flags);
      break;

#if !defined(_WIN32)
    case NchwcKernelBenchPath::DirectAsmNoPad:
      // Same argument order as the direct C++ kernel above. This entry is for
      // the no-padding direct asm microkernel path.
      MlasConvNchwcFloatKernelNeonAsm(
          buffers.input,
          buffers.filter.data(),
          buffers.output.data(),
          buffers.stride_width_bytes,
          buffers.dilation_width_bytes,
          config.filter_count,
          buffers.input_stride_bytes,
          buffers.filter_stride_bytes,
          buffers.output_stride_bytes,
          config.kernel_height,
          config.kernel_width,
          buffers.input_base,
          buffers.input_width_bytes,
          buffers.dilated_input_width_bytes,
          config.output_count_left_pad,
          config.output_count,
          config.output_count_right_pad,
          buffers.bias.data(),
          config.kernel_flags);
      break;
#endif
  }
}

void NCHWC_KERNEL_ROW(
    benchmark::State& state,
    NchwcKernelBenchPath path,
    size_t filter_count,
    size_t output_count_left_pad,
    size_t output_count,
    size_t output_count_right_pad,
    size_t kernel_height,
    size_t kernel_width) {
  if (MlasNchwcGetBlockSize() != BlockSize) {
    state.SkipWithError("Unexpected NCHWC block size for ARM NEON benchmark");
    return;
  }

  const NchwcKernelBenchConfig config{
      filter_count,
      output_count_left_pad,
      output_count,
      output_count_right_pad,
      kernel_height,
      kernel_width,
      BenchmarkKernelFlags};

  NchwcKernelBenchBuffers buffers(config);

  RunNchwcKernelBench(path, config, buffers);

  for (auto _ : state) {
    RunNchwcKernelBench(path, config, buffers);
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }

  const int64_t work_items = static_cast<int64_t>(config.filter_count * buffers.total_output_count * BlockSize);
  state.SetItemsProcessed(state.iterations() * work_items);
}

}  // namespace

// BENCHMARK_CAPTURE args after the path are:
//   filter_count, output_count_left_pad, output_count,
//   output_count_right_pad, kernel_height, kernel_width
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC1_DirectCpp_NoPad, NchwcKernelBenchPath::DirectCpp, 1, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC1_Wrapper_NoPad, NchwcKernelBenchPath::WrapperNoPad, 1, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC1_Wrapper_Padded, NchwcKernelBenchPath::WrapperPadded, 1, 1, 54, 1, 3, 3)->UseRealTime();

BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC2_DirectCpp_NoPad, NchwcKernelBenchPath::DirectCpp, 2, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC2_Wrapper_NoPad, NchwcKernelBenchPath::WrapperNoPad, 2, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC2_Wrapper_Padded, NchwcKernelBenchPath::WrapperPadded, 2, 1, 54, 1, 3, 3)->UseRealTime();

BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC3_DirectCpp_NoPad, NchwcKernelBenchPath::DirectCpp, 3, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC3_Wrapper_NoPad, NchwcKernelBenchPath::WrapperNoPad, 3, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC3_Wrapper_Padded, NchwcKernelBenchPath::WrapperPadded, 3, 1, 54, 1, 3, 3)->UseRealTime();

BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC4_DirectCpp_NoPad, NchwcKernelBenchPath::DirectCpp, 4, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC4_Wrapper_NoPad, NchwcKernelBenchPath::WrapperNoPad, 4, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC4_Wrapper_Padded, NchwcKernelBenchPath::WrapperPadded, 4, 1, 54, 1, 3, 3)->UseRealTime();

#if !defined(_WIN32)
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC1_DirectAsm_NoPad, NchwcKernelBenchPath::DirectAsmNoPad, 1, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC2_DirectAsm_NoPad, NchwcKernelBenchPath::DirectAsmNoPad, 2, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC3_DirectAsm_NoPad, NchwcKernelBenchPath::DirectAsmNoPad, 3, 0, 56, 0, 3, 3)->UseRealTime();
BENCHMARK_CAPTURE(NCHWC_KERNEL_ROW, FC4_DirectAsm_NoPad, NchwcKernelBenchPath::DirectAsmNoPad, 4, 0, 56, 0, 3, 3)->UseRealTime();
#endif

#endif
