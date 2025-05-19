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
__global__ void UnpackV(const T* input, T* output, const int token_count, const int hidden_size,
                        const int packed_seq_stride) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < token_count * hidden_size) {
    int offset = tid % hidden_size;
    int token_id = tid / hidden_size;
    int packed_i = token_id * packed_seq_stride + offset;
    output[tid] = input[packed_i];
  }
}

template <typename T>
Status LaunchUnpackCumulative(const T* input, T* output, const int token_count, const int hidden_size,
                              const int packed_seq_stride, cudaStream_t stream, const int max_threads_per_block) {
  const int threads = std::min(max_threads_per_block, token_count * hidden_size);
  const int blocks = (token_count * hidden_size + threads - 1) / threads;
  UnpackV<T><<<blocks, threads, 0, stream>>>(input, output, token_count, hidden_size, packed_seq_stride);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void RotaryEmbeddingTNH(T* output,                          // TxNxH
                                   const T* input,                     // TxNxH
                                   const T* cos_cache,                 // Mx(H/2)
                                   const T* sin_cache,                 // Mx(H/2)
                                   const int32_t* past_seqlens,        // B
                                   const int32_t* cumulative_seqlens_q,  // B+1
                                   const int head_size,
                                   const int rotary_embedding_dim,
                                   const bool interleaved,
                                   const int3 in_strides,     // TxNxH
                                   const int3 out_strides) {  // TxNxH
  // Use .x in innermost loop to access global memory efficiently

  const int b = blockIdx.y;
  const int s = blockIdx.x;
  const int n = blockIdx.z;
  const int h = threadIdx.x;

  const int sequence_length = cumulative_seqlens_q[b + 1] - cumulative_seqlens_q[b];
  if (h >= head_size || s >= sequence_length) {
    return;
  }

  const int t = cumulative_seqlens_q[b] + s; // t is the index of the token in the unpadded input/output
  const T* input_data = input + t * in_strides.x + n * in_strides.y;
  T* output_data = output + t * out_strides.x + n * out_strides.y;

  if (h >= rotary_embedding_dim) {
    output_data[h] = input_data[h];
    return;
  }

  // Cache is (M, H/2)
  const int half_rotary_embedding_dim = rotary_embedding_dim / 2;
  const int position_id = past_seqlens[b] + s;
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
  output_data[h] = input_data[h] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
}

template <typename T>
Status LaunchRotaryEmbeddingKernel(cudaStream_t stream, T* output, const T* input, const int32_t* past_seqlens,
                                   const int32_t* cumulative_seqlens_q, const T* cos_cache, const T* sin_cache,
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
      output, input, cos_cache, sin_cache, past_seqlens, cumulative_seqlens_q, head_size, rotary_embedding_dim,
      interleaved, in_strides, out_strides);
  return CUDA_CALL(cudaGetLastError());
}

template <int kBlockSize>
__global__ void GetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                       const int32_t* past_seqlens, const int batch_size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id == 0) {
    cumulative_seqlens_kv[0] = 0;
  }

  typedef cub::BlockScan<int, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Sum past_seqlens to new sequence length (which we get by subtracting cumulative_seqlens_q).
  // Then do an inclusive sum across present sequence lengths to get the cumulative sequence length
  if (id < batch_size) {
    cumulative_seqlens_kv[id + 1] = past_seqlens[id] + cumulative_seqlens_q[id + 1] - cumulative_seqlens_q[id];
    int length = cumulative_seqlens_kv[id + 1];
    BlockScan(temp_storage).InclusiveSum(length, length);
    cumulative_seqlens_kv[id + 1] = length;
  }
}

Status LaunchGetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                    const int32_t* past_seqlens, const int batch_size, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  GetCumulativeSeqlensKV<256><<<blocks, threads, 0, stream>>>(cumulative_seqlens_kv, cumulative_seqlens_q, past_seqlens,
                                                              batch_size);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void ReshapeAndCache(const T* __restrict__ key, const T* __restrict__ value, T* __restrict__ key_cache,
                                T* __restrict__ value_cache, const int* __restrict__ block_table,
                                const int* __restrict__ past_seqlens, const int* __restrict__ cumulative_seqlens_q,
                                const int batch_size, const int max_num_blocks_per_seq, const int token_count,
                                const int kv_hidden_size, const int block_size, const int key_stride,
                                const int value_stride) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= token_count * kv_hidden_size) {
    return;
  }
  const int token_id = tid / kv_hidden_size;
  const int hidden_offset = tid % kv_hidden_size;
  int batch_id = 0;
  for (int i = 0; i < batch_size; ++i) {
    if (token_id < cumulative_seqlens_q[i + 1]) {
      batch_id = i;
      break;
    }
  }
  const int token_offset = token_id - cumulative_seqlens_q[batch_id];
  const int past_length = past_seqlens[batch_id];
  const int block_id = block_table[batch_id * max_num_blocks_per_seq +  (past_length + token_offset) / block_size];
  const int block_offset = (past_length + token_offset) % block_size;

  const int key_id = token_id * key_stride + hidden_offset;
  const int value_id = token_id * value_stride + hidden_offset;
  const int dst_id = block_id * block_size * kv_hidden_size + block_offset * kv_hidden_size + hidden_offset;
  key_cache[dst_id] = key[key_id];
  value_cache[dst_id] = value[value_id];
}

template <typename T>
Status LaunchReshapeAndCache(const T* key, const T* value, T* key_cache, T* value_cache, const int* block_table,
                             const int* past_seqlens, const int* cumulative_seqlens_q, const int batch_size,
                             const int max_num_blocks_per_seq, const int token_count, const int kv_hidden_size,
                             const int block_size, const int key_stride, const int value_stride, cudaStream_t stream,
                             const int max_threads_per_block) {
  const int total_size = token_count * kv_hidden_size;
  const int threads(std::min(total_size, max_threads_per_block));
  const int blocks((total_size + threads - 1) / threads);
  ReshapeAndCache<T><<<blocks, threads, 0, stream>>>(key, value, key_cache, value_cache, block_table, past_seqlens,
                                                     cumulative_seqlens_q, batch_size, max_num_blocks_per_seq,
                                                     token_count, kv_hidden_size, block_size, key_stride, value_stride);
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
  const int q_hidden_size = parameters.hidden_size;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const float softcap = parameters.softcap;
  bool is_bf16 = std::is_same<T, BFloat16>::value;
  const int local_window_size = parameters.local_window_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int block_size = parameters.block_size;
  // The following are passed to flash api but not used by the kernel, so they can be determined heuristically
  const int max_query_len = token_count - batch_size + 1;
  const int max_seq_len = parameters.max_num_blocks_per_seq * parameters.block_size;

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
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* past_seqlens = const_cast<int*>(data.past_seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;
  ORT_RETURN_IF_ERROR(LaunchGetCumulativeSeqlensKV(cumulative_seqlens_kv, cumulative_seqlens_q, past_seqlens,
                                                   batch_size, stream));

  if (parameters.do_rotary) {
    // Will unpack Q and K in case of packed_qkv
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    const int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, q_buffer, query, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, k_buffer, key, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, kv_num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    query = q_buffer;
    key = k_buffer;
  } else if (parameters.is_packed_qkv) {
    // Only unpack Q. K and V are unpacked by ReshapeAndCache.
    auto q_buffer = data.workspace_buffer;
    const int packed_seq_stride = q_hidden_size + 2 * kv_hidden_size;
    ORT_RETURN_IF_ERROR(LaunchUnpackCumulative<T>(
        query, q_buffer, token_count, q_hidden_size, packed_seq_stride, stream, max_threads_per_block));
    query = q_buffer;
  }

  // Insert key and value into block-based KV cache
  int* block_table = const_cast<int*>(data.block_table);
  const int key_stride = parameters.is_packed_qkv && !parameters.do_rotary ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const int value_stride = parameters.is_packed_qkv ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  ORT_RETURN_IF_ERROR(LaunchReshapeAndCache<T>(key, value, data.key_cache, data.value_cache, block_table, past_seqlens,
                                               cumulative_seqlens_q, batch_size, max_num_blocks_per_seq, token_count,
                                               kv_hidden_size, block_size, key_stride, value_stride, stream,
                                               max_threads_per_block));

  // Launch kernel
  void* q = reinterpret_cast<void*>(query);
  void* key_cache = reinterpret_cast<void*>(data.key_cache);
  void* value_cache = reinterpret_cast<void*>(data.value_cache);
  void* output = reinterpret_cast<void*>(data.output);
  void* softmax_lse = reinterpret_cast<void*>(data.softmax_lse);
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
      device_prop, stream, q, key_cache, value_cache, output, cumulative_seqlens_q, cumulative_seqlens_kv,
      /*seqused_k*/ nullptr, block_table, softmax_lse, batch_size, num_heads, kv_num_heads, head_size, max_query_len,
      max_seq_len, token_count, scale, softcap, /*is_causal*/ true, is_bf16, local_window_size, max_num_blocks_per_seq,
      block_size));

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("flash attention output", data.output, token_count, num_heads, head_size);

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
