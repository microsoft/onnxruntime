// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "mlas.h"
#include "bench_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/sconv_nchwc_kernel_neon.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_ARM_NEON_NCHWC) && !defined(_WIN32)

constexpr size_t NchwcBlockSize = MLAS_PLATFORM::MLAS_NEON_NCHWC_BLOCK_SIZE;

static std::vector<std::string> ArgNamesForDirectNchwc() {
  return {"IC", "OC", "IH", "IW", "KH", "KW", "PT", "PL", "PB", "PR", "S", "D"};
}

static size_t ComputeOutputDim(size_t input, size_t kernel, size_t stride, size_t dilation, size_t pad_before,
                               size_t pad_after) {
  const size_t dilated = (kernel - 1) * dilation + 1;
  if (input + pad_before + pad_after < dilated) {
    throw std::invalid_argument("Invalid shape: input smaller than dilated kernel");
  }
  return (input + pad_before + pad_after - dilated) / stride + 1;
}

static size_t DivideRoundUp(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

static void RunDirectNchwcKernel(const size_t input_channels,
                                 const size_t output_channels,
                                 const size_t input_height,
                                 const size_t input_width,
                                 const size_t kernel_height,
                                 const size_t kernel_width,
                                 const size_t pad_top,
                                 const size_t pad_left,
                                 const size_t pad_bottom,
                                 const size_t pad_right,
                                 const size_t stride,
                                 const size_t dilation,
                                 MLAS_CONV_FLOAT_KERNEL* Kernel,
                                 const float* input,
                                 const float* filter,
                                 const float* bias,
                                 float* output) {
  // This routine drives the direct NCHWc micro-kernel exactly like the production
  // driver path: one output row and one input-channel block contribution per call.
  // The micro-kernel itself does not iterate over full output height or all IC blocks.
  const size_t output_height = ComputeOutputDim(input_height, kernel_height, stride, dilation, pad_top, pad_bottom);
  const size_t output_width = ComputeOutputDim(input_width, kernel_width, stride, dilation, pad_left, pad_right);
  const size_t kernel_size = kernel_height * kernel_width;

  const size_t input_channel_blocks = input_channels / NchwcBlockSize;
  const size_t output_channel_blocks = output_channels / NchwcBlockSize;

  const size_t stride_width_bytes = NchwcBlockSize * stride * sizeof(float);
  const size_t dilation_width_bytes = NchwcBlockSize * dilation * sizeof(float);
  const size_t filter_stride_bytes = NchwcBlockSize * input_channels * kernel_size * sizeof(float);
  const size_t output_stride_bytes = NchwcBlockSize * output_height * output_width * sizeof(float);
  const size_t input_width_bytes = NchwcBlockSize * input_width * sizeof(float);
  const size_t dilated_input_width_bytes = NchwcBlockSize * dilation * input_width * sizeof(float);
  const size_t input_stride_bytes = dilated_input_width_bytes - kernel_width * dilation_width_bytes;
  const size_t dilated_kernel_width = (kernel_width - 1) * dilation + 1;

  const size_t output_count_left_pad = std::min(output_width, DivideRoundUp(pad_left, stride));
  size_t output_count_right_pad = 0;
  while (output_count_right_pad < (output_width - output_count_left_pad)) {
    const size_t ox = output_width - output_count_right_pad - 1;
    const ptrdiff_t input_x = static_cast<ptrdiff_t>(ox * stride) - static_cast<ptrdiff_t>(pad_left);
    if (input_x + static_cast<ptrdiff_t>(dilated_kernel_width) <= static_cast<ptrdiff_t>(input_width)) {
      break;
    }
    output_count_right_pad++;
  }
  const size_t output_count = output_width - output_count_left_pad - output_count_right_pad;

  // Outer IC-block loop is required to accumulate all channel-block contributions:
  // first block initializes with bias, remaining blocks accumulate into output.
  // the production driver sets ACCUMULATE_OUTPUT for all but the first IC block,
  // and applies BIAS_ADDITION only on the final IC block.
  for (size_t icb = 0; icb < input_channel_blocks; ++icb) {
    const bool is_first_ic_block = (icb == 0);
    const bool is_last_ic_block = (icb + 1 == input_channel_blocks);
    unsigned kernel_flags = 0;
    if (!is_first_ic_block) {
      kernel_flags |= MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT;
    }
    if (is_last_ic_block && bias != nullptr) {
      kernel_flags |= MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION;
    }
    const float* ic_block_input = input + icb * input_height * input_width * NchwcBlockSize;
    const float* ic_block_filter = filter + icb * kernel_size * NchwcBlockSize * NchwcBlockSize;

    // Outer OH loop is required because the micro-kernel processes one output row.
    for (size_t oh = 0; oh < output_height; ++oh) {
      const ptrdiff_t output_origin_h = static_cast<ptrdiff_t>(oh * stride) - static_cast<ptrdiff_t>(pad_top);
      const size_t kh_start = output_origin_h < 0 ? DivideRoundUp(static_cast<size_t>(-output_origin_h), dilation) : 0;

      size_t kh_end = kernel_height;
      const ptrdiff_t input_h_limit = static_cast<ptrdiff_t>(input_height);
      if (output_origin_h + static_cast<ptrdiff_t>((kernel_height - 1) * dilation) >= input_h_limit) {
        if (output_origin_h >= input_h_limit) {
          kh_end = 0;
        } else {
          const ptrdiff_t span = input_h_limit - 1 - output_origin_h;
          kh_end = static_cast<size_t>(span / static_cast<ptrdiff_t>(dilation)) + 1;
        }
      }
      if (kh_start >= kh_end) {
        continue;
      }

      const size_t effective_kernel_height = kh_end - kh_start;
      const ptrdiff_t input_h_base = output_origin_h + static_cast<ptrdiff_t>(kh_start * dilation);

      ptrdiff_t kernel_row_index =
          input_h_base * static_cast<ptrdiff_t>(input_width) - static_cast<ptrdiff_t>(pad_left);
      if (kernel_row_index < 0) {
        kernel_row_index = 0;
      }
      const float* kernel_input_row =
          ic_block_input + static_cast<ptrdiff_t>(NchwcBlockSize) * kernel_row_index;
      const float* input_base_row =
          ic_block_input + static_cast<ptrdiff_t>(NchwcBlockSize) * (input_h_base * static_cast<ptrdiff_t>(input_width));
      const float* kernel_filter = ic_block_filter + kh_start * kernel_width * NchwcBlockSize * NchwcBlockSize;
      float* output_row = output + oh * output_width * NchwcBlockSize;

      Kernel(kernel_input_row,
             kernel_filter,
             output_row,
             stride_width_bytes,
             dilation_width_bytes,
             output_channel_blocks,
             input_stride_bytes,
             filter_stride_bytes,
             output_stride_bytes,
             effective_kernel_height,
             kernel_width,
             input_base_row,
             input_width_bytes,
             dilated_input_width_bytes,
             output_count_left_pad,
             output_count,
             output_count_right_pad,
             bias,
             kernel_flags);
    }
  }
}

static void BenchDirectNchwc(benchmark::State& state) {
  // It benchmarks the direct NCHWc kernel path only (not full graph/runtime overhead).
  // Included: kernel math, row/block driving, and padding edge handling in the driver loop.
  // Excluded: threadpool scheduling, graph transforms, model execution, and memory allocator costs.
  const size_t input_channels = static_cast<size_t>(state.range(0));
  const size_t output_channels = static_cast<size_t>(state.range(1));
  const size_t input_height = static_cast<size_t>(state.range(2));
  const size_t input_width = static_cast<size_t>(state.range(3));
  const size_t kernel_height = static_cast<size_t>(state.range(4));
  const size_t kernel_width = static_cast<size_t>(state.range(5));
  const size_t pad_top = static_cast<size_t>(state.range(6));
  const size_t pad_left = static_cast<size_t>(state.range(7));
  const size_t pad_bottom = static_cast<size_t>(state.range(8));
  const size_t pad_right = static_cast<size_t>(state.range(9));
  const size_t stride = static_cast<size_t>(state.range(10));
  const size_t dilation = static_cast<size_t>(state.range(11));

  if (input_channels == 0 || output_channels == 0 || input_height == 0 || input_width == 0 || kernel_height == 0 ||
      kernel_width == 0 || stride == 0 || dilation == 0) {
    throw std::invalid_argument("All benchmark parameters must be > 0");
  }

  if (input_channels % NchwcBlockSize != 0 || output_channels % NchwcBlockSize != 0) {
    throw std::invalid_argument("IC and OC must be multiples of MLAS NEON NCHWc block size");
  }

  const size_t output_height = ComputeOutputDim(input_height, kernel_height, stride, dilation, pad_top, pad_bottom);
  const size_t output_width = ComputeOutputDim(input_width, kernel_width, stride, dilation, pad_left, pad_right);
  const size_t kernel_size = kernel_height * kernel_width;

  const size_t input_channel_blocks = input_channels / NchwcBlockSize;
  const size_t output_channel_blocks = output_channels / NchwcBlockSize;

  const size_t input_size = input_channel_blocks * input_height * input_width * NchwcBlockSize;
  const size_t filter_size = output_channel_blocks * input_channels * kernel_size * NchwcBlockSize;
  const size_t output_size = output_channel_blocks * output_height * output_width * NchwcBlockSize;

  auto input = RandomVectorUniform(std::vector<int64_t>{static_cast<int64_t>(input_size)}, -1.0f, 1.0f);
  auto filter = RandomVectorUniform(std::vector<int64_t>{static_cast<int64_t>(filter_size)}, -1.0f, 1.0f);
  auto bias = RandomVectorUniform(std::vector<int64_t>{static_cast<int64_t>(output_channels)}, -0.5f, 0.5f);
  std::vector<float> output(output_size);

  RunDirectNchwcKernel(input_channels,
                       output_channels,
                       input_height,
                       input_width,
                       kernel_height,
                       kernel_width,
                       pad_top,
                       pad_left,
                       pad_bottom,
                       pad_right,
                       stride,
                       dilation,
                       &MlasConvNchwcFloatKernelNeon,
                       input.data(),
                       filter.data(),
                       bias.data(),
                       output.data());

  for (auto _ : state) {
    RunDirectNchwcKernel(input_channels,
                         output_channels,
                         input_height,
                         input_width,
                         kernel_height,
                         kernel_width,
                         pad_top,
                         pad_left,
                         pad_bottom,
                         pad_right,
                         stride,
                         dilation,
                         &MlasConvNchwcFloatKernelNeon,
                         input.data(),
                         filter.data(),
                         bias.data(),
                         output.data());
  }
}

void SCONV_NCHWC_DIRECT(benchmark::State& state, const char* /*dummy*/) {
  BenchDirectNchwc(state);
}

static void DirectNchwcCases(benchmark::internal::Benchmark* b) {
  b->ArgNames(ArgNamesForDirectNchwc());

  // IC, OC, IH, IW, KH, KW, PT, PL, PB, PR, S, D
  b->Args({32, 32, 192, 192, 3, 3, 1, 1, 1, 1, 1, 1});
  b->Args({32, 96, 192, 192, 3, 3, 0, 0, 1, 1, 2, 1});
  b->Args({48, 192, 96, 96, 3, 3, 0, 0, 1, 1, 2, 1});
  b->Args({48, 192, 96, 96, 3, 3, 1, 1, 1, 1, 1, 1});
  b->Args({64, 256, 48, 48, 3, 3, 1, 1, 1, 1, 1, 1});
}

BENCHMARK_CAPTURE(SCONV_NCHWC_DIRECT, DirectNchwcCases, "")->Apply(DirectNchwcCases)->UseRealTime();

#endif  // defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_ARM_NEON_NCHWC) && !defined(_WIN32)
