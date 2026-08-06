// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/compressed_attention.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float ToFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half FromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__device__ float AttentionBias(const T* bias, int64_t b_dim, int64_t n_dim,
                               int64_t s_dim, int64_t k_dim, int batch, int head,
                               int query, int key) {
  if (bias == nullptr) return 0.0f;
  const int64_t b = b_dim == 1 ? 0 : batch;
  const int64_t n = n_dim == 1 ? 0 : head;
  const int64_t s = s_dim == 1 ? 0 : query;
  const int64_t k = k_dim == 1 ? 0 : key;
  return ToFloat(bias[((b * n_dim + n) * s_dim + s) * k_dim + k]);
}

template <typename T>
__device__ const T* LocalValue(const T* current, const T* past, const int64_t* positions,
                               int batch, int query_index, int sequence_length,
                               int capacity, int head_size, int key, bool fixed_cache) {
  if (!fixed_cache) return current + (batch * capacity + key) * head_size;
  const int64_t query_position = positions[batch * sequence_length + query_index];
  const int64_t first_position = max(int64_t{0}, query_position - capacity + 1);
  const int64_t key_position = first_position + key;
  const int64_t step_start = positions[batch * sequence_length];
  if (key_position >= step_start) {
    return current + (batch * sequence_length + key_position - step_start) * head_size;
  }
  return past + (batch * capacity + key_position % capacity) * head_size;
}

template <typename T>
__global__ void CompressedAttentionKernel(
    T* output, const T* query, const T* local_kv, const T* past_local_kv,
    const int64_t* position_ids, const T* compressed_kv,
    const T* attention_bias, const int64_t* selected_indices, const T* head_sink,
    int num_heads, int sequence_length, int head_size, int local_count,
    int compressed_count, int selected_count, int sink_count, int64_t bias_b,
    int64_t bias_n, int64_t bias_s, int64_t bias_k, float scale, bool fixed_cache) {
  if (threadIdx.x != 0) return;
  const int batch = blockIdx.x / (num_heads * sequence_length);
  const int remainder = blockIdx.x % (num_heads * sequence_length);
  const int head = remainder / sequence_length;
  const int query_index = remainder % sequence_length;
  const T* q = query + ((batch * num_heads + head) * sequence_length + query_index) * head_size;
  const int visible_local_count = fixed_cache
                                      ? min(local_count, static_cast<int>(position_ids[batch * sequence_length + query_index] + 1))
                                      : local_count;
  const bool compressed_only_bias = fixed_cache && bias_k == compressed_count;
  const float sink = ToFloat(head_sink[sink_count == 1 ? 0 : head]);
  float maximum = sink;
  for (int key = 0; key < visible_local_count; ++key) {
    const T* value = LocalValue(local_kv, past_local_kv, position_ids, batch, query_index,
                                sequence_length, local_count, head_size, key, fixed_cache);
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(q[dim]) * ToFloat(value[dim]);
    const float bias = compressed_only_bias ? 0.0f : AttentionBias(
      attention_bias, bias_b, bias_n, bias_s, bias_k, batch, head, query_index, key);
    maximum = fmaxf(maximum, dot * scale + bias);
  }
  for (int index = 0; index < selected_count; ++index) {
    const int entry = selected_indices == nullptr
                          ? index
                          : static_cast<int>(selected_indices[(batch * sequence_length + query_index) * selected_count + index]);
    if (entry < 0) continue;
    const T* value = compressed_kv + (batch * compressed_count + entry) * head_size;
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(q[dim]) * ToFloat(value[dim]);
    const int bias_key = compressed_only_bias ? entry : local_count + entry;
    maximum = fmaxf(maximum, dot * scale + AttentionBias(attention_bias, bias_b, bias_n, bias_s, bias_k,
                               batch, head, query_index, bias_key));
  }
  float denominator = expf(sink - maximum);
  for (int key = 0; key < visible_local_count; ++key) {
    const T* value = LocalValue(local_kv, past_local_kv, position_ids, batch, query_index,
                                sequence_length, local_count, head_size, key, fixed_cache);
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(q[dim]) * ToFloat(value[dim]);
    const float bias = compressed_only_bias ? 0.0f : AttentionBias(
      attention_bias, bias_b, bias_n, bias_s, bias_k, batch, head, query_index, key);
    denominator += expf(dot * scale + bias - maximum);
  }
  for (int index = 0; index < selected_count; ++index) {
    const int entry = selected_indices == nullptr
                          ? index
                          : static_cast<int>(selected_indices[(batch * sequence_length + query_index) * selected_count + index]);
    if (entry < 0) continue;
    const T* value = compressed_kv + (batch * compressed_count + entry) * head_size;
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(q[dim]) * ToFloat(value[dim]);
    const int bias_key = compressed_only_bias ? entry : local_count + entry;
    denominator += expf(dot * scale + AttentionBias(attention_bias, bias_b, bias_n, bias_s, bias_k,
                             batch, head, query_index, bias_key) - maximum);
  }
  T* out = output + ((batch * num_heads + head) * sequence_length + query_index) * head_size;
  for (int dim = 0; dim < head_size; ++dim) {
    float sum = 0.0f;
    for (int key = 0; key < visible_local_count; ++key) {
      const T* value = LocalValue(local_kv, past_local_kv, position_ids, batch, query_index,
                                  sequence_length, local_count, head_size, key, fixed_cache);
      float dot = 0.0f;
      for (int d = 0; d < head_size; ++d) dot += ToFloat(q[d]) * ToFloat(value[d]);
        const float bias = compressed_only_bias ? 0.0f : AttentionBias(
          attention_bias, bias_b, bias_n, bias_s, bias_k, batch, head, query_index, key);
        const float weight = expf(dot * scale + bias - maximum) / denominator;
      sum += weight * ToFloat(value[dim]);
    }
    for (int index = 0; index < selected_count; ++index) {
      const int entry = selected_indices == nullptr
                            ? index
                            : static_cast<int>(selected_indices[(batch * sequence_length + query_index) * selected_count + index]);
      if (entry < 0) continue;
      const T* value = compressed_kv + (batch * compressed_count + entry) * head_size;
      float dot = 0.0f;
      for (int d = 0; d < head_size; ++d) dot += ToFloat(q[d]) * ToFloat(value[d]);
      const int bias_key = compressed_only_bias ? entry : local_count + entry;
      const float weight = expf(dot * scale + AttentionBias(attention_bias, bias_b, bias_n, bias_s, bias_k,
                         batch, head, query_index, bias_key) - maximum) /
                           denominator;
      sum += weight * ToFloat(value[dim]);
    }
    out[dim] = FromFloat<T>(sum);
  }
}

template <typename T>
__global__ void UpdateLocalCacheKernel(T* present, const T* current, const int64_t* positions,
                                       int sequence_length, int capacity, int head_size, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int dim = index % head_size;
  const int row = index / head_size;
  const int cache_index = row % capacity;
  const int batch = row / capacity;
  for (int token = sequence_length - 1; token >= 0; --token) {
    const int current_row = batch * sequence_length + token;
    if (positions[current_row] % capacity == cache_index) {
      present[index] = current[current_row * head_size + dim];
      return;
    }
  }
}

template <typename T>
Status LaunchCompressedAttentionKernel(
  cudaStream_t stream, T* output, T* present_local_kv, const T* query, const T* local_kv,
  const T* past_local_kv, const int64_t* position_ids,
    const T* compressed_kv, const T* attention_bias, const int64_t* selected_indices,
    const T* head_sink, int batch_size, int num_heads, int sequence_length,
    int head_size, int local_count, int compressed_count, int selected_count,
    int sink_count, int64_t bias_b, int64_t bias_n, int64_t bias_s, int64_t bias_k,
    float scale, bool fixed_cache, int max_threads_per_block) {
  CompressedAttentionKernel<T><<<batch_size * num_heads * sequence_length, 1, 0, stream>>>(
      output, query, local_kv, past_local_kv, position_ids, compressed_kv, attention_bias, selected_indices, head_sink,
      num_heads, sequence_length, head_size, local_count, compressed_count, selected_count,
      sink_count, bias_b, bias_n, bias_s, bias_k, scale, fixed_cache);
  if (fixed_cache) {
    const int count = batch_size * local_count * head_size;
    const int block = min(max_threads_per_block, 256);
    UpdateLocalCacheKernel<T><<<(count + block - 1) / block, block, 0, stream>>>(
        present_local_kv, local_kv, position_ids, sequence_length, local_count, head_size, count);
  }
  return CUDA_CALL(cudaGetLastError());
}


template Status LaunchCompressedAttentionKernel<half>(cudaStream_t, half*, half*, const half*, const half*, const half*, const int64_t*, const half*, const half*, const int64_t*, const half*, int, int, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, float, bool, int);
template Status LaunchCompressedAttentionKernel<BFloat16>(cudaStream_t, BFloat16*, BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const int64_t*, const BFloat16*, const BFloat16*, const int64_t*, const BFloat16*, int, int, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, float, bool, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
