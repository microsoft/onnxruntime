// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/nn/causal_conv_with_state_impl.h"

namespace onnxruntime::cuda {

template <typename T>
__global__ void CausalConvWithStateDecodeKernel(
    const T* input, const T* weight, const T* bias, const T* past, T* output, T* present,
    int64_t rows, int64_t channels, int64_t kernel, int64_t dilation, int64_t history, bool silu) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t channel = row % channels;
    float sum = 0;
    for (int64_t tap = 0; tap < kernel - 1; ++tap) {
      const float sample = past ? static_cast<float>(past[row * history + tap * dilation]) : 0.0f;
      sum += sample * static_cast<float>(weight[channel * kernel + tap]);
    }
    sum += static_cast<float>(input[row]) * static_cast<float>(weight[channel * kernel + kernel - 1]);
    if (bias) sum += static_cast<float>(bias[channel]);
    const T convolution = static_cast<T>(sum);
    const float value = static_cast<float>(convolution);
    output[row] = silu ? static_cast<T>(value / (1.0f + expf(-value))) : convolution;
    for (int64_t s = 0; s < history; ++s) {
      present[row * history + s] = s == history - 1
                                       ? input[row]
                                       : (past ? past[row * history + s + 1] : T{});
    }
  }
}

template <typename T, int Kernel>
__global__ void CausalConvWithStateKernel(
    const T* input, const T* weight, const T* bias, const T* past, T* output, T* present,
    int64_t rows, int64_t channels, int64_t length, int64_t kernel,
    int64_t dilation, int64_t history, bool silu) {
  if constexpr (Kernel > 0) kernel = Kernel;
  const int64_t positions = length > history ? length : history;
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t channel = row % channels;
    for (int64_t position = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < positions; position += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      if (position < length) {
        float sum = 0;
        for (int64_t tap = 0; tap < kernel; ++tap) {
          // Subtract before adding to avoid overflow of position + tap*dilation.
          const int64_t source = position - history + tap * dilation;
          const float value = source >= 0
                                  ? static_cast<float>(input[row * length + source])
                                  : (past ? static_cast<float>(past[row * history + source + history]) : 0.0f);
          sum += static_cast<float>(weight[channel * kernel + tap]) * value;
        }
        if (bias) sum += static_cast<float>(bias[channel]);
        const T convolution = static_cast<T>(sum);
        const float value = static_cast<float>(convolution);
        output[row * length + position] = silu ? static_cast<T>(value / (1.0f + expf(-value))) : convolution;
      }
      if (position < history) {
        // Equivalent to concatenated_history[length + position], without materializing it.
        const int64_t source = length - history + position;
        present[row * history + position] = source >= 0
                                                ? input[row * length + source]
                                                : (past ? past[row * history + source + history] : T{});
      }
    }
  }
}

template <typename T>
void CausalConvWithStateImpl(cudaStream_t stream, const T* input, const T* weight,
                             const T* bias, const T* past, T* output, T* present,
                             int64_t rows, int64_t channels, int64_t length,
                             int64_t kernel, int64_t dilation, int64_t history, bool silu) {
  const int64_t positions = std::max(length, history);
  if (rows == 0 || positions == 0) return;
  if (length == 1 && history <= 128) {
    const int blocks = static_cast<int>(std::min<int64_t>((rows - 1) / 128 + 1, 65535));
    CausalConvWithStateDecodeKernel<<<blocks, 128, 0, stream>>>(
        input, weight, bias, past, output, present, rows, channels, kernel, dilation, history, silu);
    return;
  }
  const int threads = positions == 1 ? 1 : 128;
  const dim3 grid(static_cast<unsigned>(std::min<int64_t>((positions - 1) / threads + 1, 65535)),
                  static_cast<unsigned>(std::min<int64_t>(rows, 65535)));
#define LAUNCH(K) CausalConvWithStateKernel<T, K><<<grid, threads, 0, stream>>>( \
    input, weight, bias, past, output, present, rows, channels, length, kernel, dilation, history, silu)
  switch (kernel) {
    case 2:
      LAUNCH(2);
      break;
    case 3:
      LAUNCH(3);
      break;
    case 4:
      LAUNCH(4);
      break;
    default:
      LAUNCH(0);
      break;
  }
#undef LAUNCH
}

#define INSTANTIATE(T)                                                                                   \
  template void CausalConvWithStateImpl<T>(cudaStream_t, const T*, const T*, const T*, const T*, T*, T*, \
                                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool);
INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(BFloat16)
#undef INSTANTIATE

}  // namespace onnxruntime::cuda
