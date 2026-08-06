// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"

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

static int ChooseBlockSize(int head_size, int max_threads) {
  int threads = std::min(head_size, max_threads);
  int power = 1;
  while (power * 2 <= threads) {
    power *= 2;
  }
  return power;
}

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
__device__ __forceinline__ T ReadCombinedToken(const T* pending, const T* current,
                                                int batch, int token, int dim,
                                                int pending_tokens, int pending_capacity,
                                                int sequence_length, int width) {
  if (token < pending_tokens) {
    return pending[(batch * pending_capacity + token) * width + dim];
  }
  return current[(batch * sequence_length + token - pending_tokens) * width + dim];
}

template <typename T>
__global__ void WriteCompressorPendingKernel(
    T* pending_kv_out, T* pending_gate_out, const T* current_kv, const T* current_gate,
    const T* past_pending_kv, const T* past_pending_gate, int sequence_length,
    int batch_size, int past_pending_tokens, int pending_capacity, int usable_tokens,
    int pending_tokens, int width, const int64_t* position_ids, int compress_rate,
    bool fixed_mode) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_capacity = fixed_mode ? pending_capacity : pending_tokens;
  const int count = batch_size * output_capacity * width;
  if (index >= count) return;
  const int dim = index % width;
  const int row = index / width;
  const int token = row % output_capacity;
  const int batch = row / output_capacity;
  if (fixed_mode) {
    past_pending_tokens = static_cast<int>(position_ids[batch * sequence_length] % compress_rate);
    usable_tokens = (past_pending_tokens + sequence_length) / compress_rate * compress_rate;
    pending_tokens = past_pending_tokens + sequence_length - usable_tokens;
    if (token >= pending_tokens) {
      pending_kv_out[index] = FromFloat<T>(0.0f);
      pending_gate_out[index] = FromFloat<T>(0.0f);
      return;
    }
  }
  const int source_token = usable_tokens + token;
  pending_kv_out[index] = ReadCombinedToken(past_pending_kv, current_kv, batch, source_token, dim,
                                             past_pending_tokens, pending_capacity, sequence_length, width);
  pending_gate_out[index] = ReadCombinedToken(past_pending_gate, current_gate, batch, source_token, dim,
                                               past_pending_tokens, pending_capacity, sequence_length, width);
}

template <typename T>
__global__ void DeepSeekV4CompressorKernel(
    T* entries, T* overlap_kv_out, T* overlap_gate_out, const T* current_kv,
    const T* current_gate, const T* past_pending_kv, const T* past_pending_gate,
    const T* past_overlap_kv, const T* past_overlap_gate, const T* position_bias,
    const T* norm_weight, const T* cos_cache, const T* sin_cache, int sequence_length,
    int past_pending_tokens, int old_entry_count, int new_entry_count, int width,
    int head_size, int compress_rate, int rotary_dim, int cos_cache_width,
    int entry_capacity, const int64_t* position_ids, float epsilon, bool is_overlap,
    bool fixed_mode) {
  const int batch = blockIdx.x / new_entry_count;
  const int window = blockIdx.x % new_entry_count;
  const int pending_capacity = fixed_mode ? compress_rate - 1 : past_pending_tokens;
  if (fixed_mode) {
    const int64_t start_position = position_ids[batch * sequence_length];
    past_pending_tokens = static_cast<int>(start_position % compress_rate);
    old_entry_count = static_cast<int>(start_position / compress_rate);
    const int batch_new_entry_count = (past_pending_tokens + sequence_length) / compress_rate;
    if (window >= batch_new_entry_count || old_entry_count + window >= entry_capacity) return;
    new_entry_count = batch_new_entry_count;
  }
  const int tid = threadIdx.x;
  const int slots = is_overlap ? 2 * compress_rate : compress_rate;
  extern __shared__ float scratch[];
  float* row = scratch;
  float* reduction = scratch + head_size;

  float local_squares = 0.0f;
  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    float max_logit = -FLT_MAX;
    for (int slot = 0; slot < slots; ++slot) {
      float logit;
      if (!is_overlap) {
        const int token = window * compress_rate + slot;
        logit = ToFloat(ReadCombinedToken(past_pending_gate, current_gate, batch, token, dim,
                                           past_pending_tokens, pending_capacity, sequence_length, width));
        logit += ToFloat(position_bias[slot * width + dim]);
      } else if (slot < compress_rate && window == 0) {
        logit = ToFloat(past_overlap_gate[(batch * compress_rate + slot) * head_size + dim]);
      } else {
        const int token = slot < compress_rate ? (window - 1) * compress_rate + slot
                                               : window * compress_rate + slot - compress_rate;
        const int source_dim = (slot < compress_rate ? 0 : head_size) + dim;
        logit = ToFloat(ReadCombinedToken(past_pending_gate, current_gate, batch, token, source_dim,
                                           past_pending_tokens, pending_capacity, sequence_length, width));
        logit += ToFloat(position_bias[(token % compress_rate) * width + source_dim]);
      }
      max_logit = fmaxf(max_logit, logit);
    }

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int slot = 0; slot < slots; ++slot) {
      float value;
      float logit;
      if (!is_overlap) {
        const int token = window * compress_rate + slot;
        value = ToFloat(ReadCombinedToken(past_pending_kv, current_kv, batch, token, dim,
                                           past_pending_tokens, pending_capacity, sequence_length, width));
        logit = ToFloat(ReadCombinedToken(past_pending_gate, current_gate, batch, token, dim,
                                           past_pending_tokens, pending_capacity, sequence_length, width));
        logit += ToFloat(position_bias[slot * width + dim]);
      } else if (slot < compress_rate && window == 0) {
        value = ToFloat(past_overlap_kv[(batch * compress_rate + slot) * head_size + dim]);
        logit = ToFloat(past_overlap_gate[(batch * compress_rate + slot) * head_size + dim]);
      } else {
        const int token = slot < compress_rate ? (window - 1) * compress_rate + slot
                                               : window * compress_rate + slot - compress_rate;
        const int source_dim = (slot < compress_rate ? 0 : head_size) + dim;
        value = ToFloat(ReadCombinedToken(past_pending_kv, current_kv, batch, token, source_dim,
                                           past_pending_tokens, pending_capacity, sequence_length, width));
        logit = ToFloat(ReadCombinedToken(past_pending_gate, current_gate, batch, token, source_dim,
                                           past_pending_tokens, pending_capacity, sequence_length, width));
        logit += ToFloat(position_bias[(token % compress_rate) * width + source_dim]);
      }
      const float weight = expf(logit - max_logit);
      weighted_sum += weight * value;
      weight_sum += weight;
    }
    row[dim] = weight_sum > 0.0f ? weighted_sum / weight_sum : 0.0f;
    local_squares += row[dim] * row[dim];
  }

  reduction[tid] = local_squares;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }

  const float inv_rms = rsqrtf(reduction[0] / static_cast<float>(head_size) + epsilon);
  T* output_entry = entries +
      (batch * entry_capacity + old_entry_count + window) * head_size;
  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    output_entry[dim] = FromFloat<T>(row[dim] * inv_rms * ToFloat(norm_weight[dim]));
  }
  __syncthreads();

  const int rotary_start = head_size - rotary_dim;
  const int position = (old_entry_count + window) * compress_rate;
  for (int pair = tid; pair < rotary_dim / 2; pair += blockDim.x) {
    const int dim = rotary_start + 2 * pair;
    const float x0 = ToFloat(output_entry[dim]);
    const float x1 = ToFloat(output_entry[dim + 1]);
    const float cosine = ToFloat(cos_cache[position * cos_cache_width + pair]);
    const float sine = ToFloat(sin_cache[position * cos_cache_width + pair]);
    output_entry[dim] = FromFloat<T>(x0 * cosine - x1 * sine);
    output_entry[dim + 1] = FromFloat<T>(x0 * sine + x1 * cosine);
  }

  if (is_overlap && window == new_entry_count - 1) {
    for (int dim = tid; dim < head_size; dim += blockDim.x) {
      for (int slot = 0; slot < compress_rate; ++slot) {
        const int token = window * compress_rate + slot;
        const int output_index = (batch * compress_rate + slot) * head_size + dim;
        overlap_kv_out[output_index] = ReadCombinedToken(
            past_pending_kv, current_kv, batch, token, dim,
            past_pending_tokens, pending_capacity, sequence_length, width);
        const float gate = ToFloat(ReadCombinedToken(
            past_pending_gate, current_gate, batch, token, dim,
            past_pending_tokens, pending_capacity, sequence_length, width));
        overlap_gate_out[output_index] = FromFloat<T>(gate + ToFloat(position_bias[slot * width + dim]));
      }
    }
  }
}

template <typename T>
Status LaunchDeepSeekV4CompressorKernel(
    cudaStream_t stream, T* entries, T* pending_kv_out, T* pending_gate_out,
    T* overlap_kv_out, T* overlap_gate_out, const T* current_kv, const T* current_gate,
    const T* past_pending_kv, const T* past_pending_gate, const T* past_overlap_kv,
    const T* past_overlap_gate, const T* position_bias, const T* norm_weight,
    const T* cos_cache, const T* sin_cache, const int64_t* position_ids,
    int batch_size, int sequence_length,
    int pending_token_count, int old_entry_count, int new_entry_count, int width,
    int head_size, int compress_rate, int rotary_dim, int cos_cache_width, int entry_capacity,
    float epsilon, bool is_overlap, bool fixed_mode, int max_threads_per_block) {
  const int total_tokens = pending_token_count + sequence_length;
  const int usable_tokens = new_entry_count * compress_rate;
  const int output_pending_tokens = total_tokens - usable_tokens;
  if (fixed_mode || output_pending_tokens > 0) {
    const int pending_capacity = fixed_mode ? compress_rate - 1 : pending_token_count;
    const int output_capacity = fixed_mode ? pending_capacity : output_pending_tokens;
    const int count = batch_size * output_capacity * width;
    const int block = std::min(max_threads_per_block, 256);
    WriteCompressorPendingKernel<T><<<(count + block - 1) / block, block, 0, stream>>>(
        pending_kv_out, pending_gate_out, current_kv, current_gate, past_pending_kv,
        past_pending_gate, sequence_length, batch_size, pending_token_count, pending_capacity,
        usable_tokens, output_pending_tokens, width, position_ids, compress_rate, fixed_mode);
  }
  if (new_entry_count > 0) {
    const int block = ChooseBlockSize(head_size, max_threads_per_block);
    const size_t shared_memory = static_cast<size_t>(head_size + block) * sizeof(float);
    DeepSeekV4CompressorKernel<T><<<batch_size * new_entry_count, block, shared_memory, stream>>>(
        entries, overlap_kv_out, overlap_gate_out, current_kv, current_gate, past_pending_kv,
        past_pending_gate, past_overlap_kv, past_overlap_gate, position_bias, norm_weight,
        cos_cache, sin_cache, sequence_length, pending_token_count, old_entry_count,
        new_entry_count, width, head_size, compress_rate, rotary_dim, cos_cache_width,
        entry_capacity, position_ids, epsilon, is_overlap, fixed_mode);
  }
  return CUDA_CALL(cudaGetLastError());
}


template Status LaunchDeepSeekV4CompressorKernel<half>(cudaStream_t, half*, half*, half*, half*, half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const half*, const int64_t*, int, int, int, int, int, int, int, int, int, int, int, float, bool, bool, int);
template Status LaunchDeepSeekV4CompressorKernel<BFloat16>(cudaStream_t, BFloat16*, BFloat16*, BFloat16*, BFloat16*, BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, const int64_t*, int, int, int, int, int, int, int, int, int, int, int, float, bool, bool, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
