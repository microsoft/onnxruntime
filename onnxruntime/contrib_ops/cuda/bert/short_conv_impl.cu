// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/short_conv_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "contrib_ops/cuda/bert/engram_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Raw (un-normalized) value at a virtual position of "past_state history followed by the current chunk".
// Missing history reads as zero, which is exactly what a fresh sequence contributes.
template <typename T>
__device__ __forceinline__ float RawAt(const T* input, const T* past_state, int64_t b, int64_t p,
                                       int64_t g, int64_t c, int64_t sequence_length,
                                       int64_t state_length, int64_t hc_mult, int64_t hidden_size) {
  if (p >= state_length) {
    const int64_t t = p - state_length;
    return to_float<T>(input[((b * sequence_length + t) * hc_mult + g) * hidden_size + c]);
  }
  if (past_state == nullptr) {
    return 0.0f;
  }
  return to_float<T>(past_state[((b * state_length + p) * hc_mult + g) * hidden_size + c]);
}

// One block per (batch, virtual position, g) row. The RMS reduction only depends on the row, so it is
// computed once here instead of being repeated by every output channel and every convolution tap.
// Recomputing it from the raw history (rather than storing normalized history) makes a chunked run
// bit-exact against a full-sequence run for every element type.
template <typename T>
__global__ void ShortConvInvRmsKernel(
    const T* input,
    const T* past_state,
    float* inv_rms,
    int64_t rows,
    int64_t sequence_length,
    int64_t state_length,
    int64_t hc_mult,
    int64_t hidden_size,
    float epsilon) {
  extern __shared__ float shared[];

  const int64_t virtual_length = state_length + sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t g = row % hc_mult;
    const int64_t p = (row / hc_mult) % virtual_length;
    const int64_t b = row / (virtual_length * hc_mult);
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float value = RawAt<T>(input, past_state, b, p, g, i, sequence_length, state_length,
                                   hc_mult, hidden_size);
      sum_sq += value * value;
    }
    sum_sq = engram_helper::BlockSum(sum_sq, shared);
    if (threadIdx.x == 0) {
      inv_rms[row] = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
    }
  }
}

// Copies the trailing state window of "past_state followed by input" through unchanged, so that a
// chunked run reproduces the full-sequence result.
template <typename T>
__global__ void ShortConvPresentStateKernel(
    const T* input,
    const T* past_state,
    T* present_state,
    int64_t total,
    int64_t sequence_length,
    int64_t state_length,
    int64_t hc_mult,
    int64_t hidden_size) {
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t c = linear % hidden_size;
    const int64_t g = (linear / hidden_size) % hc_mult;
    const int64_t slot = (linear / (hc_mult * hidden_size)) % state_length;
    const int64_t b = linear / (state_length * hc_mult * hidden_size);
    const int64_t p = sequence_length + slot;
    present_state[linear] = from_float<T>(RawAt<T>(input, past_state, b, p, g, c, sequence_length,
                                                   state_length, hc_mult, hidden_size));
  }
}

template <typename T>
__global__ void ShortConvKernel(
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    const T* past_state,
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
  const int64_t state_length = (kernel_size - 1) * dilation;
  const int64_t virtual_length = state_length + sequence_length;
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
      // Tap k is (kernel_size - 1 - k) * dilation positions back, so p is always in [0, virtual_length).
      const int64_t p = t + state_length - (kernel_size - 1 - k) * dilation;
      const int64_t virtual_row = (b * virtual_length + p) * hc_mult + g;
      const float normed = RawAt<T>(input, past_state, b, p, g, c, sequence_length, state_length,
                                    hc_mult, hidden_size) *
                           inv_rms[virtual_row] * scale;
      sum += normed * to_float<T>(weight[flat_channel * kernel_size + k]);
    }
    output[linear] = from_float<T>(apply_silu ? engram_helper::SiluFloat(sum) : sum);
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
    const T* past_state,
    float* inv_rms_workspace,
    T* output,
    T* present_state,
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

  const int64_t state_length = (kernel_size - 1) * dilation;
  const int64_t virtual_rows = batch_size * (state_length + sequence_length) * hc_mult;

  const int rms_blocks = static_cast<int>(std::min(virtual_rows, engram_helper::kMaxGridDimX));
  const size_t shared_bytes = static_cast<size_t>(engram_helper::kThreads) * sizeof(float);
  ShortConvInvRmsKernel<T><<<rms_blocks, engram_helper::kThreads, shared_bytes, stream>>>(
      input, past_state, inv_rms_workspace, virtual_rows, sequence_length, state_length, hc_mult,
      hidden_size, epsilon);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  ShortConvKernel<T><<<engram_helper::GridSize(total), engram_helper::kThreads, 0, stream>>>(
      input, weight, norm_scale, bias, past_state, inv_rms_workspace, output, total, sequence_length,
      hc_mult, hidden_size, kernel_size, dilation, apply_silu);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  const int64_t present_total = batch_size * state_length * hc_mult * hidden_size;
  if (present_state == nullptr || present_total == 0) {
    return Status::OK();
  }
  ShortConvPresentStateKernel<T><<<engram_helper::GridSize(present_total), engram_helper::kThreads, 0, stream>>>(
      input, past_state, present_state, present_total, sequence_length, state_length, hc_mult,
      hidden_size);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_SHORT_CONV(T)                                                                \
  template Status LaunchShortConvKernel<T>(cudaStream_t, const T*, const T*, const T*, const T*, \
                                           const T*, float*, T*, T*, int64_t, int64_t, int64_t,  \
                                           int64_t, int64_t, int64_t, float, bool);

INSTANTIATE_SHORT_CONV(float)
INSTANTIATE_SHORT_CONV(half)
INSTANTIATE_SHORT_CONV(__nv_bfloat16)

#undef INSTANTIATE_SHORT_CONV

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
