// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xqa_loader.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

#include <cuda_bf16.h>
#include <cfloat>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace H512 {
namespace {

constexpr int kHeadSize = 512;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// Each warp owns an interleaved subset of visible tokens; a second kernel merges all partitions.
constexpr int kSplitBlocks = kGQAXqaH512Partitions / kWarps;
constexpr int kPartialCount = kWarps * kSplitBlocks;

template <typename T>
__device__ float ToFloat(T value);

template <>
__device__ float ToFloat<half>(half value) {
  return __half2float(value);
}

template <>
__device__ float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T FromFloat(float value);

template <>
__device__ half FromFloat<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 FromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void FusedGqaDecode512Partial(const T* query,
                                         const T* key_cache,
                                         const T* value_cache,
                                         float* partial_max,
                                         float* partial_sum,
                                         float* partial_output,
                                         int num_heads,
                                         int kv_num_heads,
                                         int max_seq_len,
                                         float scale,
                                         int local_window_size,
                                         bool is_bsnh,
                                         const int* past_seq_lens) {
  const int batch = blockIdx.y;
  const int query_head = blockIdx.x;
  const int kv_head = query_head / (num_heads / kv_num_heads);
  const int sequence_length = past_seq_lens[batch] + 1;
  const int first_token = local_window_size > 0 && sequence_length > local_window_size
                              ? sequence_length - local_window_size
                              : 0;
  const size_t query_base = (static_cast<size_t>(batch) * num_heads + query_head) * kHeadSize;
  constexpr int kValuesPerThread = kHeadSize / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int partition = blockIdx.z * kWarps + warp;
  const size_t partial_index = ((static_cast<size_t>(batch) * num_heads + query_head) * kPartialCount) + partition;

  float query_values[kValuesPerThread];
  float accumulators[kValuesPerThread] = {};
  // Lanes span contiguous head dimensions so cache loads coalesce; retain Q across the token loop.
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    query_values[i] = ToFloat(query[query_base + lane + i * 32]);
  }

  float row_max = -FLT_MAX;
  float row_sum = 0.0f;
  for (int token = first_token + partition; token < sequence_length; token += kPartialCount) {
    const size_t cache_base = is_bsnh
                                  ? ((static_cast<size_t>(batch) * max_seq_len + token) * kv_num_heads + kv_head) * kHeadSize
                                  : ((static_cast<size_t>(batch) * kv_num_heads + kv_head) * max_seq_len + token) * kHeadSize;
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      partial += query_values[i] * ToFloat(key_cache[cache_base + lane + i * 32]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    const float score = __shfl_sync(0xffffffff, partial, 0) * scale;
    // Online softmax rescales both the denominator and weighted V sum when the running maximum grows.
    float new_max;
    float alpha;
    float beta;
    if (lane == 0) {
      new_max = fmaxf(row_max, score);
      alpha = expf(row_max - new_max);
      beta = expf(score - new_max);
    }
    new_max = __shfl_sync(0xffffffff, new_max, 0);
    alpha = __shfl_sync(0xffffffff, alpha, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      accumulators[i] = accumulators[i] * alpha + ToFloat(value_cache[cache_base + lane + i * 32]) * beta;
    }
    row_sum = row_sum * alpha + beta;
    row_max = new_max;
  }

  // Empty partitions retain a zero denominator and zero output, contributing nothing during merge.
  if (lane == 0) {
    partial_max[partial_index] = row_max;
    partial_sum[partial_index] = row_sum;
  }
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    partial_output[partial_index * kHeadSize + lane + i * 32] = accumulators[i];
  }
}

template <typename T>
__global__ void FusedGqaDecode512Merge(T* output,
                                       const float* partial_max,
                                       const float* partial_sum,
                                       const float* partial_output,
                                       int num_heads,
                                       const float* attention_sinks) {
  const int batch = blockIdx.y;
  const int query_head = blockIdx.x;
  const size_t query_base = (static_cast<size_t>(batch) * num_heads + query_head) * kHeadSize;
  const size_t partial_base = (static_cast<size_t>(batch) * num_heads + query_head) * kPartialCount;
  __shared__ float merge_correction[kPartialCount];
  __shared__ float merged_sum_shared;

  // Bring every partition to one softmax maximum before normalizing. An attention sink contributes
  // only to the denominator (its value vector is zero), and must be counted once across partitions.
  if (threadIdx.x == 0) {
    float merged_max = attention_sinks == nullptr ? -FLT_MAX : attention_sinks[query_head];
#pragma unroll
    for (int i = 0; i < kPartialCount; ++i) {
      merged_max = fmaxf(merged_max, partial_max[partial_base + i]);
    }
    float merged_sum = attention_sinks == nullptr ? 0.0f : expf(attention_sinks[query_head] - merged_max);
#pragma unroll
    for (int i = 0; i < kPartialCount; ++i) {
      merge_correction[i] = expf(partial_max[partial_base + i] - merged_max);
      merged_sum += partial_sum[partial_base + i] * merge_correction[i];
    }
    merged_sum_shared = merged_sum;
  }
  __syncthreads();

  for (int dimension = threadIdx.x; dimension < kHeadSize; dimension += kThreads) {
    float merged_value = 0.0f;
#pragma unroll
    for (int i = 0; i < kPartialCount; ++i) {
      merged_value += partial_output[(partial_base + i) * kHeadSize + dimension] * merge_correction[i];
    }
    output[query_base + dimension] = FromFloat<T>(merged_value / merged_sum_shared);
  }
}

}  // namespace

template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const int local_window_size,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* attention_sinks,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size) {
  ORT_RETURN_IF(device_prop.major < 8, "H512 fused GQA requires Ampere or newer.");
  ORT_RETURN_IF(head_size != kHeadSize, "H512 fused GQA received head_size ", head_size, ".");
  ORT_RETURN_IF(kv_num_heads <= 0 || num_heads % kv_num_heads != 0,
                "H512 fused GQA requires an integral query-to-KV head ratio.");
  ORT_RETURN_IF(kv_quant_type != XqaQuantType::kNone,
                "H512 fused GQA does not support a quantized KV cache.");
  const size_t partial_entries = static_cast<size_t>(batch_size) * num_heads * kPartialCount;
  const size_t required_workspace = partial_entries * (2 + kHeadSize) * sizeof(float);
  ORT_RETURN_IF(workspace == nullptr || workspace_size < required_workspace,
                "H512 fused GQA workspace is too small. Expected ", required_workspace,
                " bytes, got ", workspace_size, ".");
  // FP32 scratch is laid out as [maxima][denominators][512-element weighted sums], one entry per partition.
  float* partial_max = reinterpret_cast<float*>(workspace);
  float* partial_sum = partial_max + partial_entries;
  float* partial_output = partial_sum + partial_entries;

  FusedGqaDecode512Partial<T><<<dim3(num_heads, batch_size, kSplitBlocks), kThreads, 0, stream>>>(
      reinterpret_cast<const T*>(query), reinterpret_cast<const T*>(key_cache),
      reinterpret_cast<const T*>(value_cache), partial_max, partial_sum, partial_output,
      num_heads, kv_num_heads, max_seq_len, scale, local_window_size, is_bsnh, past_seq_lens);
  FusedGqaDecode512Merge<T><<<dim3(num_heads, batch_size), kThreads, 0, stream>>>(
      reinterpret_cast<T*>(output), partial_max, partial_sum, partial_output, num_heads, attention_sinks);
  return CUDA_CALL(cudaGetLastError());
}

size_t GetWorkspaceSize(int batch_size, int num_heads) {
  return static_cast<size_t>(batch_size) * num_heads * kPartialCount * (2 + kHeadSize) * sizeof(float);
}

size_t GetXQAKernelSmemBytes(int group_size) {
  ORT_UNUSED_PARAMETER(group_size);
  // No dynamic shared memory is requested; the merge kernel's small shared arrays are statically allocated.
  return 0;
}

template Status LaunchXQAKernelImpl<half>(
    const cudaDeviceProp&, cudaStream_t, const void*, const void*, const void*, void*, int, int, int, int, int,
    float, int, bool, const int*, const float*, XqaQuantType, void*, size_t);

template Status LaunchXQAKernelImpl<__nv_bfloat16>(
    const cudaDeviceProp&, cudaStream_t, const void*, const void*, const void*, void*, int, int, int, int, int,
    float, int, bool, const int*, const float*, XqaQuantType, void*, size_t);

}  // namespace H512

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime