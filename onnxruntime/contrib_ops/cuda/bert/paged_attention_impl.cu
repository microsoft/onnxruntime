/*
 The implementation of this file is based on our Group Query Attention impl.cu file,
 which is based on Multi-Head Attention impl.cu file,
 which is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modifications:
// (1) unidirectional mask (causal)
// (2) use flash attention kernel from (https://github.com/Dao-AILab/flash-attention)
// (3) support different number of heads for Q and KV
// (4) support rotary embedding
// (5) support both packed and unpacked QKV
// (6) support block-based KV Cache / Paged Attention
// (7) etc.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include <cublas_v2.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

////////// Auxiliary Kernels

template <typename T>
__global__ void UnpackQKVCumulative(const T* packed_qkv, T* unpacked_qkv, const int token_count, const int num_heads,
                                    const int kv_num_heads, const int head_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= token_count * (num_heads + 2 * kv_num_heads) * head_size) {
    return;
  }
  const int q_hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int in_seq_stride = q_hidden_size + 2 * kv_hidden_size;

  int packed_i;
  if (tid < token_count * q_hidden_size) {
    const int token_id = tid / q_hidden_size;
    const int offset = tid % q_hidden_size;
    packed_i = token_id * in_seq_stride + offset;
  } else if (tid < token_count * (q_hidden_size + kv_hidden_size)) {
    const int id = tid - token_count * q_hidden_size;
    const int token_id = id / kv_hidden_size;
    const int offset = id % kv_hidden_size;
    packed_i = token_id * in_seq_stride + q_hidden_size + offset;
  } else if (tid < token_count * (q_hidden_size + 2 * kv_hidden_size)) {
    const int id = tid - token_count * (q_hidden_size + kv_hidden_size);
    const int token_id = id / kv_hidden_size;
    const int offset = id % kv_hidden_size;
    packed_i = token_id * in_seq_stride + q_hidden_size + kv_hidden_size + offset;
  }
  unpacked_qkv[tid] = packed_qkv[packed_i];
}

// Since QKV is unpacked into a single workspace buffer, this is similar to a transpose
template <typename T>
Status LaunchUnpackQKVCumulative(const T* packed_qkv, T* unpacked_qkv, const int token_count, const int num_heads,
                                 const int kv_num_heads, const int head_size, cudaStream_t stream,
                                 const int max_threads_per_block) {
  const int threads = max_threads_per_block;
  const int blocks = (token_count * (num_heads + 2 * kv_num_heads) * head_size + threads - 1) / threads;
  UnpackQKVCumulative<T><<<blocks, threads, 0, stream>>>(packed_qkv, unpacked_qkv, token_count, num_heads, kv_num_heads,
                                                         head_size);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void UnpackV(const T* packed_v, T* unpacked_v, const int token_count, const int kv_hidden_size,
                        const int packed_seq_stride) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < token_count * kv_hidden_size) {
    int offset = tid % kv_hidden_size;
    int token_id = tid / kv_hidden_size;
    int packed_i = token_id * packed_seq_stride + offset;
    unpacked_v[tid] = packed_v[packed_i];
  }
}

template <typename T>
Status LaunchUnpackV(const T* packed_v, T* unpacked_v, const int token_count, const int kv_hidden_size,
                   const int packed_seq_stride, cudaStream_t stream, const int max_threads_per_block) {
  const int threads = std::min(max_threads_per_block, token_count * kv_hidden_size);
  const int blocks = (token_count * kv_hidden_size + threads - 1) / threads;
  UnpackV<T><<<blocks, threads, 0, stream>>>(packed_v, unpacked_v, token_count, kv_hidden_size, packed_seq_stride);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void RotaryEmbeddingTNH(T* output,                          // TxNxH
                                   const T* input,                     // TxNxH
                                   const T* cos_cache,                 // Mx(H/2)
                                   const T* sin_cache,                 // Mx(H/2)
                                   const int32_t* seqlens,             // B
                                   const int32_t* cumulative_seqlens,  // B+1
                                   const int head_size,
                                   const int rotary_embedding_dim,
                                   const bool interleaved,
                                   const int3 in_strides,     // TxNxH
                                   const int3 out_strides) {  // TxNxH
  // B = batch size, S = sequence length, N = num heads, H = head size, M = max sequence length
  // Use .x in innermost loop to access global memory efficiently

  const int b = blockIdx.y;
  const int s = blockIdx.x;
  const int n = blockIdx.z;
  const int h = threadIdx.x;

  const int sequence_length = cumulative_seqlens[b + 1] - cumulative_seqlens[b];
  if (h >= head_size || s >= sequence_length) {
    return;
  }

  const int t = cumulative_seqlens[b] + s; // t is the index of the token in the unpadded input/output
  const T* input_data = input + t * in_strides.x + n * in_strides.y;
  T* output_data = output + t * out_strides.x + n * out_strides.y;

  if (h >= rotary_embedding_dim) {
    output_data[h] = input_data[h];
    return;
  }

  // Cache is (M, H/2)
  const int half_rotary_embedding_dim = rotary_embedding_dim / 2;
  const int position_id = seqlens[b] + s;
  const int cache_offset = position_id * half_rotary_embedding_dim;
  const T* cos_data = cos_cache + cache_offset;
  const T* sin_data = sin_cache + cache_offset;

  int cache_idx = 0;
  T sign = 0;
  int j = 0;
  if (interleaved) {
    cache_idx = (h / 2) % half_rotary_embedding_dim;
    sign = (h % 2 == 0) ? -1 : 1;
    j = (h % 2 == 0) ? h + 1 : h - 1;  // i - sign
  } else {
    cache_idx = h % half_rotary_embedding_dim;
    sign = (h < half_rotary_embedding_dim) ? -1 : 1;
    j = (h + half_rotary_embedding_dim) % rotary_embedding_dim;
  }
  // output_data[h] = cos_data[cache_idx] + sin_data[cache_idx];
  output_data[h] = input_data[h] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
}

template <typename T>
Status LaunchRotaryEmbeddingKernel(cudaStream_t stream, T* output, const T* input, const int32_t* seqlens,
                                   const int32_t* cumulative_sequence_lengths, const T* cos_cache, const T* sin_cache,
                                   const int batch_size, const int max_seqlen_q, const int num_heads,
                                   const int head_size, const int rotary_embedding_dim, const bool interleaved,
                                   const int in_seq_stride, const int max_threads_per_block) {
  ORT_ENFORCE(head_size <= max_threads_per_block, "Rotary embedding dim must be <= max_threads_per_block");
  int3 in_strides = {in_seq_stride <= 0 ? num_heads * head_size : in_seq_stride, head_size, 1};
  int3 out_strides = {num_heads * head_size, head_size, 1};
  int tpb = (head_size + 31) / 32 * 32;
  const dim3 grid(max_seqlen_q, batch_size, num_heads);
  const dim3 block(tpb);
  RotaryEmbeddingTNH<<<grid, block, 0, stream>>>(
      output, input, cos_cache, sin_cache, seqlens, cumulative_sequence_lengths, head_size, rotary_embedding_dim,
      interleaved, in_strides, out_strides);
  return CUDA_CALL(cudaGetLastError());
}

template <int kBlockSize>
__global__ void GetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_sequence_length,
                                       const int32_t* seqlens, const int batch_size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id == 0) {
    cumulative_seqlens_kv[0] = 0;
  }

  typedef cub::BlockScan<int, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Sum seqlens(past sequence length) to new sequence length (which we get by subtracting cumulative_sequence_lengths).
  // Then do an inclusive sum across present sequence lengths to get the cumulative sequence length
  if (id < batch_size) {
    cumulative_seqlens_kv[id + 1] = seqlens[id] + cumulative_sequence_length[id + 1] - cumulative_sequence_length[id];
    int length = cumulative_seqlens_kv[id + 1];
    BlockScan(temp_storage).InclusiveSum(length, length);
    cumulative_seqlens_kv[id + 1] = length;
  }
}

Status LaunchGetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_sequence_length,
                                    const int32_t* seqlens, const int batch_size, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  GetCumulativeSeqlensKV<256><<<blocks, threads, 0, stream>>>(cumulative_seqlens_kv, cumulative_sequence_length, seqlens,
                                                         batch_size);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void ReshapeAndCache(const T* __restrict__ key, const T* __restrict__ value, T* __restrict__ key_cache,
                                T* __restrict__ value_cache, const int* __restrict__ slot_mappings,
                                const int token_count, const int kv_hidden_size, const int block_size) {
  const int token_id = blockIdx.x;
  const int slot_id = slot_mappings[token_id];
  const int block_id = slot_id / block_size;
  const int block_offset = slot_id % block_size;

  for (int i = threadIdx.x; i < kv_hidden_size; i += blockDim.x) {
    const int src_id = token_id * kv_hidden_size + i; // id in key/value
    const int dst_id = block_id * block_size * kv_hidden_size + block_offset * kv_hidden_size + i; // id in cache
    key_cache[dst_id] = key[src_id];
    value_cache[dst_id] = value[src_id];
  }
}

template <typename T>
Status LaunchReshapeAndCache(const T* key, const T* value, T* key_cache, T* value_cache, const int* slot_mappings,
                             const int token_count, const int kv_hidden_size, const int block_size, cudaStream_t stream,
                             const int max_threads_per_block) {
  const dim3 blocks(token_count);
  const dim3 threads(std::min(kv_hidden_size, max_threads_per_block));
  ReshapeAndCache<T><<<blocks, threads, 0, stream>>>(key, value, key_cache, value_cache, slot_mappings, token_count,
                                                     kv_hidden_size, block_size);
  return CUDA_CALL(cudaGetLastError());
}

////////// Launch Kernels

#if USE_FLASH_ATTENTION
template <typename T>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T>& data,
    float scale) {
  // Get parameters
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int token_count = parameters.token_count;
  const int max_query_len = parameters.sequence_length;
  const int max_seq_len = parameters.total_sequence_length;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const float softcap = parameters.softcap;
  bool is_bf16 = std::is_same<T, BFloat16>::value;
  const int local_window_size = parameters.local_window_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int block_size = parameters.block_size;

  T* query = const_cast<T*>(data.query);
  T* key;
  T* value;
  if (!parameters.is_packed_qkv) {
    key = const_cast<T*>(data.key);
    value = const_cast<T*>(data.value);
  } else {
    key = reinterpret_cast<T*>(query) + static_cast<size_t>(num_heads * head_size);
    value = reinterpret_cast<T*>(key) + static_cast<size_t>(kv_num_heads * head_size);
  }

  // Calculate cumulative present sequence length in cumulative_seqlens_kv
  int* cumulative_sequence_length = const_cast<int*>(data.cumulative_sequence_length);
  int* seqlens = const_cast<int*>(data.seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;
  ORT_RETURN_IF_ERROR(LaunchGetCumulativeSeqlensKV(cumulative_seqlens_kv, cumulative_sequence_length, seqlens,
                                                   batch_size, stream));

  if (parameters.do_rotary) { // Will also perform unpacking if packed_qkv
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, q_buffer, query, seqlens, cumulative_sequence_length, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, k_buffer, key, seqlens, cumulative_sequence_length, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, kv_num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    query = q_buffer;
    key = k_buffer;
    if (parameters.is_packed_qkv) {
      auto v_buffer = data.workspace_buffer + token_count * (num_heads + kv_num_heads) * head_size;
      ORT_RETURN_IF_ERROR(LaunchUnpackV<T>(
          value, v_buffer, token_count, kv_hidden_size, packed_seq_stride, stream, max_threads_per_block));
      value = v_buffer;
    }
  } else if (parameters.is_packed_qkv) {
    auto qkv_buffer = data.workspace_buffer;
    ORT_RETURN_IF_ERROR(LaunchUnpackQKVCumulative<T>(
        query, qkv_buffer, token_count, num_heads, kv_num_heads, head_size, stream, max_threads_per_block));
    query = qkv_buffer;
    key = qkv_buffer + token_count * num_heads * head_size;
    value = qkv_buffer + token_count * (num_heads + kv_num_heads) * head_size;
  }

  // Insert key and value into block-based KV cache
  ORT_RETURN_IF_ERROR(LaunchReshapeAndCache<T>(key, value, data.key_cache, data.value_cache, data.slot_mappings,
                                               token_count, kv_hidden_size, block_size, stream, max_threads_per_block));

  // Launch kernel
  void* q = reinterpret_cast<void*>(query);
  void* key_cache = reinterpret_cast<void*>(data.key_cache);
  void* value_cache = reinterpret_cast<void*>(data.value_cache);
  void* output = reinterpret_cast<void*>(data.output);
  void* softmax_lse = reinterpret_cast<void*>(data.softmax_lse);
  int* block_table = const_cast<int*>(data.block_table);
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
      device_prop, stream, q, key_cache, value_cache, output, cumulative_sequence_length, cumulative_seqlens_kv,
      /*seqused_k*/ nullptr, block_table, softmax_lse, batch_size, num_heads, kv_num_heads, head_size, max_query_len,
      max_seq_len, scale, softcap, /*is_causal*/ true, is_bf16, local_window_size, max_num_blocks_per_seq, block_size));

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, num_heads, head_size);

  return Status::OK();
}
#endif

////////// API Functions

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& /*cublas*/,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    return FlashAttention(device_prop, stream, parameters, data, scale);
  }
#endif

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unfused Paged Attention not implemented.");
}

template struct PagedAttentionData<half>;
template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<half>& data);

template struct PagedAttentionData<BFloat16>;
template Status QkvToContext<BFloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<BFloat16>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
