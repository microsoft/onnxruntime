// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/engram_gate_impl.h"

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

// One block per (token, g) row. The gate is a scalar for the whole row, so it is reduced once by the
// block and then applied to every output channel, instead of being recomputed by each channel.
template <typename T>
__global__ void EngramGateKernel(
    const T* embeddings,
    const T* hidden_states,
    const T* key_weight,
    const T* key_bias,
    const T* value_weight,
    const T* value_bias,
    const T* key_norm_scale,
    const T* query_norm_scale,
    const T* conv_norm_scale,
    T* output,
    T* output_normed,
    int64_t rows,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon) {
  extern __shared__ float shared[];

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t g = row % hc_mult;
    const int64_t token = row / hc_mult;
    const T* embedding_row = embeddings + token * embedding_size;
    const T* hidden_row = hidden_states + row * hidden_size;
    const T* key_weight_g = key_weight + g * embedding_size * hidden_size;
    const T* key_scale_g = key_norm_scale + g * hidden_size;
    const T* query_scale_g = query_norm_scale + g * hidden_size;
    const T* key_bias_g = key_bias == nullptr ? nullptr : key_bias + g * hidden_size;

    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    float dot_numerator = 0.0f;

    for (int64_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
      float key = key_bias_g == nullptr ? 0.0f : to_float<T>(key_bias_g[d]);
      for (int64_t e = 0; e < embedding_size; ++e) {
        key += to_float<T>(embedding_row[e]) * to_float<T>(key_weight_g[e * hidden_size + d]);
      }
      const float query = to_float<T>(hidden_row[d]);
      key_sum_sq += key * key;
      query_sum_sq += query * query;
      dot_numerator += key * to_float<T>(key_scale_g[d]) * query * to_float<T>(query_scale_g[d]);
    }

    key_sum_sq = kernel_helper::BlockSum(key_sum_sq, shared);
    query_sum_sq = kernel_helper::BlockSum(query_sum_sq, shared);
    dot_numerator = kernel_helper::BlockSum(dot_numerator, shared);

    const float key_inv_rms = rsqrtf(key_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float query_inv_rms = rsqrtf(query_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float dot = dot_numerator * key_inv_rms * query_inv_rms / sqrtf(static_cast<float>(hidden_size));
    const float gate = kernel_helper::SigmoidFloat(kernel_helper::EngramGateArg(dot));

    T* output_row = output + row * hidden_size;
    float gated_sum_sq = 0.0f;
    for (int64_t c = threadIdx.x; c < hidden_size; c += blockDim.x) {
      float value = value_bias == nullptr ? 0.0f : to_float<T>(value_bias[c]);
      for (int64_t e = 0; e < embedding_size; ++e) {
        value += to_float<T>(embedding_row[e]) * to_float<T>(value_weight[e * hidden_size + c]);
      }
      const float gated_value = gate * value;
      gated_sum_sq += gated_value * gated_value;
      output_row[c] = from_float<T>(gated_value);
    }

    if (output_normed != nullptr) {
      gated_sum_sq = kernel_helper::BlockSum(gated_sum_sq, shared);
      const float normed_inv_rms = rsqrtf(gated_sum_sq / static_cast<float>(hidden_size) + epsilon);
      const T* conv_scale_g = conv_norm_scale + g * hidden_size;
      T* output_normed_row = output_normed + row * hidden_size;
      for (int64_t c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        output_normed_row[c] =
            from_float<T>(to_float<T>(output_row[c]) * normed_inv_rms * to_float<T>(conv_scale_g[c]));
      }
    }
  }
}

}  // namespace

template <typename T>
Status LaunchEngramGateKernel(
    cudaStream_t stream,
    const T* embeddings,
    const T* hidden_states,
    const T* key_weight,
    const T* key_bias,
    const T* value_weight,
    const T* value_bias,
    const T* key_norm_scale,
    const T* query_norm_scale,
    const T* conv_norm_scale,
    T* output,
    T* output_normed,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon) {
  const int64_t rows = batch_size * sequence_length * hc_mult;
  if (rows == 0 || hidden_size == 0) {
    return Status::OK();
  }
  const int blocks = static_cast<int>(std::min(rows, kernel_helper::kMaxGridDimX));
  const size_t shared_bytes = static_cast<size_t>(kernel_helper::kThreads) * sizeof(float);
  EngramGateKernel<T><<<blocks, kernel_helper::kThreads, shared_bytes, stream>>>(
      embeddings, hidden_states, key_weight, key_bias, value_weight, value_bias, key_norm_scale,
      query_norm_scale, conv_norm_scale, output, output_normed, rows, hc_mult, hidden_size, embedding_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchEngramGateKernel<float>(cudaStream_t, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, float);
template Status LaunchEngramGateKernel<half>(cudaStream_t, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t, float);
template Status LaunchEngramGateKernel<__nv_bfloat16>(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, float);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
