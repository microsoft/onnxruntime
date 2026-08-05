// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/deepseek_v4_indexer.h"

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
__global__ void DeepSeekV4IndexerKernel(
    int64_t* selected_indices, T* queries, const T* head_weights, const T* entries,
    const int64_t* position_ids, const T* cos_cache, const T* sin_cache,
    int sequence_length, int num_heads, int head_size, int entry_count, int index_topk,
    int compress_rate, int rotary_dim, int cos_cache_width) {
  if (threadIdx.x != 0) return;
  const int batch = blockIdx.x / sequence_length;
  const int query_index = blockIdx.x % sequence_length;
  const int row = batch * sequence_length + query_index;
  const int64_t position = position_ids[row];
  const int rotary_start = head_size - rotary_dim;
  for (int head = 0; head < num_heads; ++head) {
    T* query = queries + (row * num_heads + head) * head_size;
    for (int pair = 0; pair < rotary_dim / 2; ++pair) {
      const int dim = rotary_start + pair * 2;
      const float x0 = ToFloat(query[dim]);
      const float x1 = ToFloat(query[dim + 1]);
      const float cosine = ToFloat(cos_cache[position * cos_cache_width + pair]);
      const float sine = ToFloat(sin_cache[position * cos_cache_width + pair]);
      query[dim] = FromFloat<T>(x0 * cosine - x1 * sine);
      query[dim + 1] = FromFloat<T>(x0 * sine + x1 * cosine);
    }
  }

  int64_t* output = selected_indices + row * index_topk;
  for (int rank = 0; rank < index_topk; ++rank) output[rank] = -1;
  const int visible = min(entry_count, static_cast<int>((position + 1) / compress_rate));
  const float dot_scale = rsqrtf(static_cast<float>(head_size));
  const float head_scale = rsqrtf(static_cast<float>(num_heads));
  for (int entry = 0; entry < visible; ++entry) {
    float score = 0.0f;
    const T* key = entries + (batch * entry_count + entry) * head_size;
    for (int head = 0; head < num_heads; ++head) {
      const T* query = queries + (row * num_heads + head) * head_size;
      float dot = 0.0f;
      for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(query[dim]) * ToFloat(key[dim]);
      score += ToFloat(head_weights[row * num_heads + head]) * head_scale * fmaxf(0.0f, dot * dot_scale);
    }
    int insert = min(index_topk, entry + 1) - 1;
    if (entry >= index_topk) {
      const int64_t tail = output[index_topk - 1];
      float tail_score = 0.0f;
      const T* tail_key = entries + (batch * entry_count + tail) * head_size;
      for (int head = 0; head < num_heads; ++head) {
        const T* query = queries + (row * num_heads + head) * head_size;
        float dot = 0.0f;
        for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(query[dim]) * ToFloat(tail_key[dim]);
        tail_score += ToFloat(head_weights[row * num_heads + head]) * head_scale *
                      fmaxf(0.0f, dot * dot_scale);
      }
      if (tail_score > score || (tail_score == score && tail < entry)) continue;
    }
    while (insert > 0) {
      const int64_t previous = output[insert - 1];
      float previous_score = 0.0f;
      const T* previous_key = entries + (batch * entry_count + previous) * head_size;
      for (int head = 0; head < num_heads; ++head) {
        const T* query = queries + (row * num_heads + head) * head_size;
        float dot = 0.0f;
        for (int dim = 0; dim < head_size; ++dim) dot += ToFloat(query[dim]) * ToFloat(previous_key[dim]);
        previous_score += ToFloat(head_weights[row * num_heads + head]) * head_scale *
                          fmaxf(0.0f, dot * dot_scale);
      }
      if (previous_score > score || (previous_score == score && previous < entry)) break;
      if (insert < index_topk) output[insert] = previous;
      --insert;
    }
    if (insert < index_topk) output[insert] = entry;
  }
}

template <typename T>
Status LaunchDeepSeekV4IndexerKernel(
    cudaStream_t stream, int64_t* selected_indices, T* queries, const T* head_weights,
    const T* entries, const int64_t* position_ids, const T* cos_cache, const T* sin_cache,
    int batch_size, int sequence_length, int num_heads, int head_size, int entry_count,
    int index_topk, int compress_rate, int rotary_dim, int cos_cache_width,
    int) {
  DeepSeekV4IndexerKernel<T><<<batch_size * sequence_length, 1, 0, stream>>>(
      selected_indices, queries, head_weights, entries, position_ids, cos_cache, sin_cache,
      sequence_length, num_heads, head_size, entry_count, index_topk, compress_rate,
      rotary_dim, cos_cache_width);
  return CUDA_CALL(cudaGetLastError());
}


template Status LaunchDeepSeekV4IndexerKernel<half>(cudaStream_t, int64_t*, half*, const half*, const half*, const int64_t*, const half*, const half*, int, int, int, int, int, int, int, int, int, int); template Status LaunchDeepSeekV4IndexerKernel<BFloat16>(cudaStream_t, int64_t*, BFloat16*, const BFloat16*, const BFloat16*, const int64_t*, const BFloat16*, const BFloat16*, int, int, int, int, int, int, int, int, int, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
