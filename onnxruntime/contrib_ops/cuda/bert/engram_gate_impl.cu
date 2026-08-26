// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/engram_gate_impl.h"

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

template <typename T>
__global__ void EngramGateKernel(
    const T* key,
    const T* query,
    const T* value,
    const T* key_norm_scale,
    const T* query_norm_scale,
    const T* conv_norm_scale,
    T* output,
    T* output_normed,
    int64_t rows,
    int64_t hc_mult,
    int64_t hidden_size,
    float epsilon) {
  extern __shared__ float shared[];

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t g = row % hc_mult;
    const int64_t token = row / hc_mult;
    const T* key_row = key + row * hidden_size;
    const T* query_row = query + row * hidden_size;
    const T* value_row = value + token * hidden_size;
    const T* key_scale_g = key_norm_scale + g * hidden_size;
    const T* query_scale_g = query_norm_scale + g * hidden_size;
    const T* conv_scale_g = conv_norm_scale == nullptr ? nullptr : conv_norm_scale + g * hidden_size;

    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    float dot_numerator = 0.0f;

    for (int64_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
      const float key_value = to_float<T>(key_row[d]);
      const float query_value = to_float<T>(query_row[d]);
      key_sum_sq += key_value * key_value;
      query_sum_sq += query_value * query_value;
      dot_numerator += key_value * to_float<T>(key_scale_g[d]) * query_value * to_float<T>(query_scale_g[d]);
    }

    engram_helper::BlockSum3(&key_sum_sq, &query_sum_sq, &dot_numerator, shared);

    const float key_inv_rms = rsqrtf(key_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float query_inv_rms = rsqrtf(query_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float dot = dot_numerator * key_inv_rms * query_inv_rms / sqrtf(static_cast<float>(hidden_size));
    const float gate = engram_helper::SigmoidFloat(engram_helper::EngramGateArg(dot));

    T* output_row = output + row * hidden_size;
    float gated_sum_sq = 0.0f;
    for (int64_t c = threadIdx.x; c < hidden_size; c += blockDim.x) {
      const float gated_value = gate * to_float<T>(value_row[c]);
      gated_sum_sq += gated_value * gated_value;
      output_row[c] = from_float<T>(gated_value);
    }

    if (output_normed != nullptr) {
      shared[threadIdx.x] = gated_sum_sq;
      __syncthreads();
      for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
      }
      const float normed_inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + epsilon);
      T* output_normed_row = output_normed + row * hidden_size;
      for (int64_t c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        output_normed_row[c] = from_float<T>(to_float<T>(output_row[c]) * normed_inv_rms * to_float<T>(conv_scale_g[c]));
      }
      __syncthreads();
    }
  }
}

}  // namespace

template <typename T>
Status LaunchEngramGateKernel(
    cudaStream_t stream,
    const T* key,
    const T* query,
    const T* value,
    const T* key_norm_scale,
    const T* query_norm_scale,
    const T* conv_norm_scale,
    T* output,
    T* output_normed,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    float epsilon) {
  const int64_t rows = batch_size * sequence_length * hc_mult;
  if (rows == 0 || hidden_size == 0) {
    return Status::OK();
  }
  const int blocks = static_cast<int>(std::min(rows, engram_helper::kMaxGridDimX));
  const size_t shared_bytes = 3 * static_cast<size_t>(engram_helper::kThreads) * sizeof(float);
  EngramGateKernel<T><<<blocks, engram_helper::kThreads, shared_bytes, stream>>>(
      key, query, value, key_norm_scale, query_norm_scale, conv_norm_scale, output, output_normed,
      rows, hc_mult, hidden_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_ENGRAM_GATE(T)                                                                            \
  template Status LaunchEngramGateKernel<T>(cudaStream_t, const T*, const T*, const T*, const T*, const T*,  \
                                            const T*, T*, T*, int64_t, int64_t, int64_t, int64_t, float);

INSTANTIATE_ENGRAM_GATE(float)
INSTANTIATE_ENGRAM_GATE(half)
INSTANTIATE_ENGRAM_GATE(__nv_bfloat16)

#undef INSTANTIATE_ENGRAM_GATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
