// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/engram_gate_impl.h"

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
    T* output,
    int64_t total,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon) {
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t c = linear % hidden_size;
    const int64_t g = (linear / hidden_size) % hc_mult;
    const int64_t token = linear / (hc_mult * hidden_size);
    const T* embedding_row = embeddings + token * embedding_size;
    const T* hidden_row = hidden_states + (token * hc_mult + g) * hidden_size;

    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    float dot_numerator = 0.0f;
    float value = value_bias == nullptr ? 0.0f : to_float<T>(value_bias[c]);

    for (int64_t i = 0; i < embedding_size; ++i) {
      value += to_float<T>(embedding_row[i]) * to_float<T>(value_weight[i * hidden_size + c]);
    }

    for (int64_t d = 0; d < hidden_size; ++d) {
      float key = key_bias == nullptr ? 0.0f : to_float<T>(key_bias[g * hidden_size + d]);
      for (int64_t e = 0; e < embedding_size; ++e) {
        key += to_float<T>(embedding_row[e]) *
               to_float<T>(key_weight[(g * embedding_size + e) * hidden_size + d]);
      }
      const float query = to_float<T>(hidden_row[d]);
      key_sum_sq += key * key;
      query_sum_sq += query * query;
      dot_numerator += key * to_float<T>(key_norm_scale[g * hidden_size + d]) *
                       query * to_float<T>(query_norm_scale[g * hidden_size + d]);
    }

    const float key_inv_rms = rsqrtf(key_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float query_inv_rms = rsqrtf(query_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float dot = dot_numerator * key_inv_rms * query_inv_rms / sqrtf(static_cast<float>(hidden_size));
    const float gate_arg = copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot);
    output[linear] = from_float<T>(SigmoidFloat(gate_arg) * value);
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
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon) {
  const int64_t total = batch_size * sequence_length * hc_mult * hidden_size;
  if (total == 0) {
    return Status::OK();
  }
  EngramGateKernel<T><<<GridSize(total), kThreads, 0, stream>>>(
      embeddings, hidden_states, key_weight, key_bias, value_weight, value_bias, key_norm_scale,
      query_norm_scale, output, total, hc_mult, hidden_size, embedding_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchEngramGateKernel<float>(cudaStream_t, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, float);
template Status LaunchEngramGateKernel<half>(cudaStream_t, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t, float);
template Status LaunchEngramGateKernel<__nv_bfloat16>(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, float);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
