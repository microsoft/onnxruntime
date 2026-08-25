// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/short_conv_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxGridDimX = 65535;

inline int GridSize(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(std::min(blocks, kMaxGridDimX));
}

__device__ __forceinline__ float SigmoidFloat(float x) {
  return x > 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

__device__ __forceinline__ float SiluFloat(float x) {
  return x * SigmoidFloat(x);
}

template <typename T>
__global__ void ShortConvKernel(
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    float epsilon,
    bool apply_silu) {
  const int64_t channels = hc_mult * hidden_size;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t c = linear % hidden_size;
    const int64_t g = (linear / hidden_size) % hc_mult;
    const int64_t t = (linear / channels) % sequence_length;
    const int64_t b = linear / (sequence_length * channels);
    const int64_t flat_channel = g * hidden_size + c;

    float sum = bias == nullptr ? 0.0f : to_float<T>(bias[flat_channel]);
    for (int64_t k = 0; k < kernel_size; ++k) {
      const int64_t source_t = t - (kernel_size - 1 - k) * dilation;
      if (source_t < 0) {
        continue;
      }

      const int64_t row_base = ((b * sequence_length + source_t) * hc_mult + g) * hidden_size;
      float sum_sq = 0.0f;
      for (int64_t i = 0; i < hidden_size; ++i) {
        const float value = to_float<T>(input[row_base + i]);
        sum_sq += value * value;
      }
      const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
      const float normed = to_float<T>(input[row_base + c]) * inv_rms *
                           to_float<T>(norm_scale[g * hidden_size + c]);
      sum += normed * to_float<T>(weight[flat_channel * kernel_size + k]);
    }
    output[linear] = from_float<T>(apply_silu ? SiluFloat(sum) : sum);
  }
}

}  // namespace

template <typename T>
Status LaunchShortConvKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    float epsilon,
    bool apply_silu) {
  const int64_t total = batch_size * sequence_length * hc_mult * hidden_size;
  if (total == 0) {
    return Status::OK();
  }
  ShortConvKernel<T><<<GridSize(total), kThreads, 0, stream>>>(
      input, weight, norm_scale, bias, output, total, sequence_length, hc_mult, hidden_size,
      kernel_size, dilation, epsilon, apply_silu);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchShortConvKernel<float>(cudaStream_t, const float*, const float*, const float*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);
template Status LaunchShortConvKernel<half>(cudaStream_t, const half*, const half*, const half*, const half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);
template Status LaunchShortConvKernel<__nv_bfloat16>(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
