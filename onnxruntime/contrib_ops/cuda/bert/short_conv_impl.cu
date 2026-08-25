// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/short_conv_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "contrib_ops/cuda/bert/kernel_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// One block per (batch, t, g) row. The RMS reduction only depends on the row, so it is computed
// once here instead of being repeated by every output channel and every convolution tap.
template <typename T>
__global__ void ShortConvInvRmsKernel(
    const T* input,
    float* inv_rms,
    int64_t rows,
    int64_t hidden_size,
    float epsilon) {
  extern __shared__ float shared[];

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const T* input_row = input + row * hidden_size;
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float value = to_float<T>(input_row[i]);
      sum_sq += value * value;
    }
    sum_sq = kernel_helper::BlockSum(sum_sq, shared);
    if (threadIdx.x == 0) {
      inv_rms[row] = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
    }
  }
}

template <typename T>
__global__ void ShortConvKernel(
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    const float* inv_rms,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
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
    const float scale = to_float<T>(norm_scale[flat_channel]);

    float sum = bias == nullptr ? 0.0f : to_float<T>(bias[flat_channel]);
    for (int64_t k = 0; k < kernel_size; ++k) {
      const int64_t source_t = t - (kernel_size - 1 - k) * dilation;
      if (source_t < 0) {
        continue;
      }

      const int64_t source_row = (b * sequence_length + source_t) * hc_mult + g;
      const float normed = to_float<T>(input[source_row * hidden_size + c]) * inv_rms[source_row] * scale;
      sum += normed * to_float<T>(weight[flat_channel * kernel_size + k]);
    }
    output[linear] = from_float<T>(apply_silu ? kernel_helper::SiluFloat(sum) : sum);
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
    float* inv_rms_workspace,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    float epsilon,
    bool apply_silu) {
  const int64_t rows = batch_size * sequence_length * hc_mult;
  const int64_t total = rows * hidden_size;
  if (total == 0) {
    return Status::OK();
  }

  const int rms_blocks = static_cast<int>(std::min(rows, kernel_helper::kMaxGridDimX));
  const size_t shared_bytes = static_cast<size_t>(kernel_helper::kThreads) * sizeof(float);
  ShortConvInvRmsKernel<T><<<rms_blocks, kernel_helper::kThreads, shared_bytes, stream>>>(
      input, inv_rms_workspace, rows, hidden_size, epsilon);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  ShortConvKernel<T><<<kernel_helper::GridSize(total), kernel_helper::kThreads, 0, stream>>>(
      input, weight, norm_scale, bias, inv_rms_workspace, output, total, sequence_length, hc_mult,
      hidden_size, kernel_size, dilation, apply_silu);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchShortConvKernel<float>(cudaStream_t, const float*, const float*, const float*, const float*, float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);
template Status LaunchShortConvKernel<half>(cudaStream_t, const half*, const half*, const half*, const half*, float*, half*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);
template Status LaunchShortConvKernel<__nv_bfloat16>(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float*, __nv_bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
