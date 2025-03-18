// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/sparse_attention_impl.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_common.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_v1_api.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v2/sparse_attention_v2_api.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Convert total_seq_len_k (total key sequence length excluding paddings) to position_ids for Prompt
__global__ void PositionIdsPrompt(const int32_t* total_seq_len_k,
                                  int64_t* position_ids,
                                  int sequence_length,
                                  int batch_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < batch_size * sequence_length) {
    int b = tid / sequence_length;
    int s = tid % sequence_length;
    if (s < total_seq_len_k[b]) {
      position_ids[tid] = s;
    } else {
      // padding
      position_ids[tid] = 1;
    }
  }
}

// Convert total_seq_len_k (total key sequence length excluding paddings) to position_ids for Token Generation
__global__ void PositionIdsToken(const int32_t* total_seq_len_k,
                                 int64_t* position_ids,
                                 int batch_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < batch_size) {
    position_ids[tid] = total_seq_len_k[tid] - 1;
  }
}

// Convert total_seq_len_k (total key sequence length excluding paddings) to position_ids
Status FillPositionIds(contrib::SparseAttentionParameters& parameters,
                       const int32_t* total_seq_len_k,
                       int64_t* position_ids,
                       cudaStream_t stream,
                       const int max_threads_per_block) {
  const int sequence_length = parameters.sequence_length;
  const int batch_size = parameters.batch_size;
  const int bs = batch_size * sequence_length;

  int threads = max_threads_per_block;
  if (bs <= 64) {
    threads = 64;
  } else if (bs <= 128) {
    threads = 128;
  } else if (bs <= 256) {
    threads = 256;
  } else if (bs <= 512) {
    threads = 512;
  }
  const int blocks = (bs + threads - 1) / threads;

  if (parameters.sequence_length == parameters.total_sequence_length) {  // prompt
    PositionIdsPrompt<<<blocks, threads, 0, stream>>>(total_seq_len_k, position_ids, sequence_length, batch_size);
  } else {
    PositionIdsToken<<<blocks, threads, 0, stream>>>(total_seq_len_k, position_ids, batch_size);
  }

  return CUDA_CALL(cudaGetLastError());
}

// Concat new key and value (BSNH format) to kv buffer (BNSH format) in place.
template <typename T>
Status LaunchConcatKVInPlace(contrib::SparseAttentionParameters& parameters,
                             SparseAttentionData<T>& data,
                             const void* new_key,
                             const void* new_value,
                             bool is_new_kv_bnsh_format,
                             cudaStream_t stream,
                             const int max_threads_per_block) {
  constexpr bool is_past_kv_bnsh_format = true;
  return LaunchConcatKVInPlace(parameters.batch_size,
                               parameters.kv_num_heads,
                               parameters.head_size,
                               parameters.max_cache_sequence_length,
                               nullptr,
                               data.seqlens_k_total,
                               parameters.sequence_length,
                               reinterpret_cast<const T*>(new_key),
                               reinterpret_cast<const T*>(new_value),
                               data.present_key,
                               data.present_value,
                               is_past_kv_bnsh_format,
                               is_new_kv_bnsh_format,
                               stream,
                               max_threads_per_block);
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<T>& data) {
  cudaStream_t stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  const void* query;
  const void* key;
  const void* value;

  DUMP_TENSOR_INIT();

  if (!parameters.is_packed_qkv) {
    static_assert(sizeof(T) == 2);
    ORT_RETURN_IF_ERROR(Transpose_BSNH_to_BNSH(
        batch_size, sequence_length, num_heads, head_size,
        reinterpret_cast<const half*>(data.query), reinterpret_cast<half*>(data.transposed_q_buffer),
        stream, max_threads_per_block));
    query = reinterpret_cast<const void*>(data.transposed_q_buffer);
    key = reinterpret_cast<const void*>(data.key);
    value = reinterpret_cast<const void*>(data.value);
  } else {
    size_t q_size = static_cast<size_t>(batch_size * sequence_length * num_heads * head_size);
    size_t k_size = static_cast<size_t>(batch_size * sequence_length * kv_num_heads * head_size);
    auto q = reinterpret_cast<T*>(data.unpacked_qkv_buffer);
    auto k = reinterpret_cast<T*>(data.unpacked_qkv_buffer + q_size);
    auto v = reinterpret_cast<T*>(data.unpacked_qkv_buffer + q_size + k_size);

    Status status = LaunchUnpackQKV<T, LAYOUT_BNSH>(data.query, q, k, v, num_heads, kv_num_heads, head_size,
                                                    sequence_length, batch_size, stream, max_threads_per_block);
    if (status != Status::OK()) {
      return status;
    }

    query = reinterpret_cast<const void*>(q);
    key = reinterpret_cast<const void*>(k);
    value = reinterpret_cast<const void*>(v);
  }

  constexpr bool q_layout = LAYOUT_BNSH;
  bool kv_layout = parameters.is_packed_qkv ? LAYOUT_BNSH : LAYOUT_BSNH;

#if DUMP_TENSOR_LEVEL > 0
  DUMP_TENSOR("query (BNSH)", reinterpret_cast<const T*>(query), batch_size, num_heads, sequence_length, head_size);

  if (LAYOUT_BNSH == kv_layout) {
    DUMP_TENSOR("key (BNSH)", reinterpret_cast<const T*>(key), batch_size, kv_num_heads, sequence_length, head_size);
    DUMP_TENSOR("value (BNSH)", reinterpret_cast<const T*>(value), batch_size, kv_num_heads, sequence_length, head_size);
  } else {
    DUMP_TENSOR("key (BSNH)", reinterpret_cast<const T*>(key), batch_size, sequence_length, kv_num_heads, head_size);
    DUMP_TENSOR("value (BSNH)", reinterpret_cast<const T*>(value), batch_size, sequence_length, kv_num_heads, head_size);
  }
#endif

  if (parameters.do_rotary) {
    size_t bsh = static_cast<size_t>(parameters.batch_size * parameters.sequence_length * parameters.head_size);
    size_t q_size = bsh * static_cast<size_t>(parameters.num_heads);
    size_t k_size = bsh * static_cast<size_t>(parameters.kv_num_heads);
    auto q_buffer = reinterpret_cast<T*>(data.rotary_buffer);
    auto k_buffer = q_buffer + q_size;
    auto position_ids_buff = reinterpret_cast<int64_t*>(k_buffer + k_size);
    ORT_RETURN_IF_ERROR(FillPositionIds(parameters, data.seqlens_k_total, position_ids_buff, stream,
                                        max_threads_per_block));

    DUMP_TENSOR("position_ids", position_ids_buff, batch_size, sequence_length);

    // Launch rotary embedding kernel. This requires separated Q, K and V
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(stream, q_buffer, reinterpret_cast<const T*>(query),
                                                       position_ids_buff, data.cos_cache, data.sin_cache,
                                                       parameters.batch_size, parameters.sequence_length,
                                                       parameters.num_heads, parameters.head_size,
                                                       parameters.rotary_dim, parameters.max_rotary_sequence_length,
                                                       /*position_ids_format*/ 1, parameters.rotary_interleaved,
                                                       max_threads_per_block, q_layout));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(stream, k_buffer, reinterpret_cast<const T*>(key),
                                                       position_ids_buff, data.cos_cache, data.sin_cache,
                                                       parameters.batch_size, parameters.sequence_length,
                                                       parameters.kv_num_heads, parameters.head_size,
                                                       parameters.rotary_dim, parameters.max_rotary_sequence_length,
                                                       /*position_ids_format*/ 1, parameters.rotary_interleaved,
                                                       max_threads_per_block, kv_layout));
    query = reinterpret_cast<const void*>(q_buffer);
    key = reinterpret_cast<const void*>(k_buffer);

#if DUMP_TENSOR_LEVEL > 0
    DUMP_TENSOR("query after rotary", reinterpret_cast<const T*>(query),
                batch_size, num_heads, sequence_length, head_size);
    if (LAYOUT_BNSH == kv_layout) {
      DUMP_TENSOR("key after rotary", reinterpret_cast<const T*>(key),
                  batch_size, kv_num_heads, sequence_length, head_size);
    } else {
      DUMP_TENSOR("key after rotary", reinterpret_cast<const T*>(key),
                  batch_size, sequence_length, kv_num_heads, head_size);
    }
#endif
  }

  // Concat new key and value to kv buffers (in BNSH format) in place
  ORT_ENFORCE(parameters.past_present_share_buffer);
  ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(
      parameters, data, key, value, kv_layout, stream, max_threads_per_block));

  // TODO: only dump to total sequence length instead of max sequence length.
#if DUMP_TENSOR_LEVEL > 0
  DUMP_TENSOR("key cache", data.present_key, batch_size, kv_num_heads,
              parameters.max_cache_sequence_length, head_size);
  DUMP_TENSOR("value cache", data.present_value, batch_size, kv_num_heads,
              parameters.max_cache_sequence_length, head_size);

  DUMP_TENSOR("csr_col_indices",
              data.kernel_layout.csr_col_indices,
              data.kernel_layout.num_layout,
              parameters.stride_col_indices);

  DUMP_TENSOR("csr_row_indices",
              data.kernel_layout.csr_row_indices,
              data.kernel_layout.num_layout,
              parameters.stride_row_indices);

  printf(
      "batch_size=%d, sequence_length=%d, num_heads=%d, kv_num_heads=%d head_size=%d, "
      "total_sequence_length=%d, max_sequence_length=%d max_cache_sequence_length=%d scale=%f block_size=%d "
      "row_stride=%d col_stride=%d num_layout=%d\n",
      parameters.batch_size,
      parameters.sequence_length,
      parameters.num_heads,
      parameters.kv_num_heads,
      parameters.head_size,
      parameters.total_sequence_length,
      parameters.max_sequence_length,
      parameters.max_cache_sequence_length,
      parameters.scale,
      data.kernel_layout.block_size,
      parameters.stride_row_indices,
      parameters.stride_col_indices,
      data.kernel_layout.num_layout);
#endif

  int sm = device_prop.major * 10 + device_prop.minor;
  if (data.use_v2_kernel) {
    sparse_attention_v2::SparseAttentionParams params(
        ort_stream,
        sm,
        data.output,
        reinterpret_cast<const void*>(query),
        reinterpret_cast<const void*>(data.present_key),
        reinterpret_cast<const void*>(data.present_value),
        q_layout == LAYOUT_BNSH,
        parameters.batch_size,
        parameters.sequence_length,
        parameters.num_heads,
        parameters.kv_num_heads,
        parameters.head_size,
        parameters.total_sequence_length,
        parameters.max_cache_sequence_length,
        parameters.scale,
        data.kernel_layout.block_size,       // kernel_block_size
        data.kernel_layout.csr_row_indices,  // shape (num_layout, stride_row_indices)
        data.kernel_layout.csr_col_indices,  // shape (num_layout, stride_col_indices)
        parameters.stride_row_indices,
        parameters.stride_col_indices,
        data.kernel_layout.num_layout,
        data.active_q_blocks,
        data.q_batch_starts,
        data.q_batch_ends,
        data.k_batch_starts,
        data.k_batch_ends,
        data.q_batch_ids,
        data.q_start_sids);

    if constexpr (std::is_same<T, BFloat16>::value) {
      ORT_RETURN_IF_ERROR(sparse_attention_v2::run_sparse_attention_bf16(params));
    } else {
      ORT_RETURN_IF_ERROR(sparse_attention_v2::run_sparse_attention_fp16(params));
    }
  } else {
    sparse_attention_v1::SparseAttentionParams params(
        ort_stream,
        sm,
        data.output,
        reinterpret_cast<const void*>(query),
        reinterpret_cast<const void*>(data.present_key),
        reinterpret_cast<const void*>(data.present_value),
        q_layout == LAYOUT_BNSH,
        parameters.batch_size,
        parameters.sequence_length,
        parameters.num_heads,
        parameters.kv_num_heads,
        parameters.head_size,
        parameters.total_sequence_length,
        parameters.max_cache_sequence_length,
        parameters.scale,
        data.kernel_layout.block_size,       // kernel_block_size
        data.kernel_layout.csr_row_indices,  // (num_layout, stride_row_indices)
        data.kernel_layout.csr_col_indices,  // (num_layout, stride_row_indices)
        parameters.stride_row_indices,
        parameters.stride_col_indices,
        data.kernel_layout.num_layout);

    if constexpr (std::is_same<T, BFloat16>::value) {
      ORT_RETURN_IF_ERROR(sparse_attention_v1::run_sparse_attention_bf16(params));
    } else {
      ORT_RETURN_IF_ERROR(sparse_attention_v1::run_sparse_attention_fp16(params));
    }
  }

  DUMP_TENSOR("output", reinterpret_cast<const T*>(data.output), batch_size, sequence_length, num_heads, head_size);

  return Status::OK();
}

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<half>& data);

template Status QkvToContext<BFloat16>(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<BFloat16>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
