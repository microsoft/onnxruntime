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

// // Kernel to unpack qkv from packed qkv
// template <typename T, bool output_bnsh>
// __global__ void UnpackQKV(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
//                           const int kv_num_heads, const int head_size, const int sequence_length,
//                           const int batch_size) {
//   const int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   int d = (num_heads + 2 * kv_num_heads) * head_size;
//   const int qkv_size = batch_size * sequence_length * d;
//   const int q_hidden = num_heads * head_size;
//   const int k_hidden = kv_num_heads * head_size;
//   if (tid < qkv_size) {
//     int b = tid / (d * sequence_length);
//     int s = (tid % (d * sequence_length)) / d;
//     int offset = tid % d;
//     if (output_bnsh) {  // output BNSH
//       int head_count = kv_num_heads;
//       T* unpacked;
//       if (offset < q_hidden) {
//         unpacked = unpacked_q;
//         head_count = num_heads;
//       } else if (offset < q_hidden + k_hidden) {
//         unpacked = unpacked_k;
//         offset -= q_hidden;
//       } else {
//         unpacked = unpacked_v;
//         offset -= (q_hidden + k_hidden);
//       }
//       int n = offset / head_size;
//       int h = offset % head_size;

//       int unpacked_i = INDEX_4D(head_count, sequence_length, head_size, b, n, s, h);
//       unpacked[unpacked_i] = packed_qkv[tid];
//     } else {  // output BSNH
//       if (offset < q_hidden) {
//         int unpacked_i = b * sequence_length * num_heads * head_size + s * num_heads * head_size + offset;
//         unpacked_q[unpacked_i] = packed_qkv[tid];
//       } else if (offset < q_hidden + k_hidden) {
//         int unpacked_i = b * sequence_length * kv_num_heads * head_size +
//                          s * kv_num_heads * head_size + (offset - q_hidden);
//         unpacked_k[unpacked_i] = packed_qkv[tid];
//       } else {
//         int unpacked_i = b * sequence_length * kv_num_heads * head_size +
//                          s * kv_num_heads * head_size + (offset - q_hidden - k_hidden);
//         unpacked_v[unpacked_i] = packed_qkv[tid];
//       }
//     }
//   }
// }

// // Unpack packed qkv
// template <typename T, bool output_bnsh>
// Status LaunchUnpackQKV(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
//                        const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size,
//                        cudaStream_t stream, const int max_threads_per_block) {
//   const int threads = max_threads_per_block;
//   const int blocks = (batch_size * sequence_length * (num_heads + 2 * kv_num_heads) * head_size + threads - 1) / threads;
//   UnpackQKV<T, output_bnsh><<<blocks, threads, 0, stream>>>(
//       packed_qkv, unpacked_q, unpacked_k, unpacked_v, num_heads, kv_num_heads, head_size, sequence_length, batch_size);
//   return CUDA_CALL(cudaGetLastError());
// }

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

  void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
  // void* key;
  // void* value;

  // TODO(aciddelgado): must we unpack or can we stride?
  // Get key and value pointers
  // if (!parameters.is_packed_qkv) {
  //   key = reinterpret_cast<void*>(const_cast<T*>(data.key));
  //   value = reinterpret_cast<void*>(const_cast<T*>(data.value));
  // } else {
  //   const size_t key_offset = static_cast<size_t>(num_heads * head_size);
  //   const size_t value_offset = static_cast<size_t>(kv_num_heads * head_size);
  //   key = reinterpret_cast<T*>(query) + key_offset;
  //   value = reinterpret_cast<T*>(key) + value_offset;
  // }

  // Calculate cumulative present sequence length in cumulative_seqlens_kv
  int* cumulative_sequence_length = const_cast<int*>(data.cumulative_sequence_length);
  int* seqlens = const_cast<int*>(data.seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;
  ORT_RETURN_IF_ERROR(LaunchGetCumulativeSeqlensKV(cumulative_seqlens_kv, cumulative_sequence_length, seqlens,
                                                   batch_size, stream));

  // Insert key and value into block-based KV cache
  ORT_RETURN_IF_ERROR(LaunchReshapeAndCache<T>(data.key, data.value, data.key_cache, data.value_cache,
                                               data.slot_mappings, token_count, kv_hidden_size, block_size, stream,
                                               max_threads_per_block));
  void* key_cache = reinterpret_cast<void*>(data.key_cache);
  void* value_cache = reinterpret_cast<void*>(data.value_cache);

  // TODO(aciddelgado): Perform rotary embedding
  // void* cos_cache = reinterpret_cast<void*>(const_cast<T*>(data.cos_cache));
  // void* sin_cache = reinterpret_cast<void*>(const_cast<T*>(data.sin_cache));

  // Launch kernel
  void* output = reinterpret_cast<void*>(data.output);
  void* softmax_lse = reinterpret_cast<void*>(data.softmax_lse);
  int* block_table = const_cast<int*>(data.block_table);
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
      device_prop, stream, query, key_cache, value_cache, output, cumulative_sequence_length, cumulative_seqlens_kv,
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

// template Status LaunchUnpackQKVCumulative<half>(
//     const half* packed_qkv, half* unpacked_q, half* unpacked_k, half* unpacked_v, const int num_heads,
//     const int kv_num_heads, const int head_size, const int token_count, cudaStream_t stream,
//     const int max_threads_per_block);

// template Status LaunchUnpackQKVCumulative<BFloat16>(
//     const BFloat16* packed_qkv, BFloat16* unpacked_q, BFloat16* unpacked_k, BFloat16* unpacked_v, const int num_heads,
//     const int kv_num_heads, const int head_size, const int token_count, cudaStream_t stream,
//     const int max_threads_per_block);

template Status LaunchReshapeAndCache<half>(
    const half* key, const half* value, half* key_cache, half* value_cache, const int* slot_mappings,
    const int token_count, const int kv_hidden_size, const int block_size, cudaStream_t stream,
    const int max_threads_per_block);

template Status LaunchReshapeAndCache<BFloat16>(
    const BFloat16* key, const BFloat16* value, BFloat16* key_cache, BFloat16* value_cache, const int* slot_mappings,
    const int token_count, const int kv_hidden_size, const int block_size, cudaStream_t stream,
    const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
