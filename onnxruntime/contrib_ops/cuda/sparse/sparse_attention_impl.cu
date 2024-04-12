// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/sparse_attention_impl.h"
#include "contrib_ops/cuda/sparse/sparse_attention_tunable.h"
#include "contrib_ops/cuda/sparse/block_mask.h"
#include "contrib_ops/cpu/utils/console_dumper.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"

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
  if (bs < 64) {
    threads = 64;
  } else if (bs < 128) {
    threads = 128;
  } else if (bs < 256) {
    threads = 256;
  } else if (bs < 512) {
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

// Kernel to append new kv to kv buffer in place
template <typename T>
__global__ void ConcatKVInPlace(const int max_sequence_length,
                                T* kv_buff,
                                const T* new_kv,
                                const int* total_seq_len_k,
                                const bool is_bsnh) {  // refers to kv buff; otherwise bnsh
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int present_batch_stride = max_sequence_length * num_heads * H;
  const int present_row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : max_sequence_length * H;

  // kv_buff:     BTNH or BNTH with buffered memory for new
  // new_kv:      BLNH

  const int past_seq_len = total_seq_len_k[b] - new_seqlen;

  int out_offset = b * present_batch_stride + (s + past_seq_len) * present_row_stride + n * present_head_stride + h;
  // Note: new KV always BSNH
  const int new_batch_stride = new_seqlen * num_heads * H;
  const int new_row_stride = num_heads * H;
  const int new_head_stride = H;
  const int in_offset = b * new_batch_stride + s * new_row_stride + n * new_head_stride + h;
  kv_buff[out_offset] = new_kv[in_offset];
}

template <typename T>
__global__ void ConcatKVInPlaceLarge(const int max_sequence_length,
                                     const int H,
                                     const int num_heads,
                                     T* kv_buff,
                                     const T* new_kv,
                                     const int* total_seq_len_k,
                                     const bool is_bsnh) {  // refers to kv buff; otherwise bnsh
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int new_seqlen = gridDim.y;

    const int present_batch_stride = max_sequence_length * num_heads * H;
    const int present_row_stride = is_bsnh ? num_heads * H : H;
    const int present_head_stride = is_bsnh ? H : max_sequence_length * H;

    // kv_buff:     BTNH or BNTH with buffered memory for new
    // new_kv:      BLNH

    const int past_seq_len = total_seq_len_k[b] - new_seqlen;

    int out_offset = b * present_batch_stride + (s + past_seq_len) * present_row_stride + n * present_head_stride + h;
    // Note: new KV always BSNH
    const int new_batch_stride = new_seqlen * num_heads * H;
    const int new_row_stride = num_heads * H;
    const int new_head_stride = H;
    const int in_offset = b * new_batch_stride + s * new_row_stride + n * new_head_stride + h;
    kv_buff[out_offset] = new_kv[in_offset];
  }
}

// Concat new to kv buffer in place
template <typename T>
Status LaunchConcatKVInPlace(contrib::SparseAttentionParameters& parameters,
                             SparseAttentionData<T>& data,
                             const void* new_key,
                             const void* new_value,
                             cudaStream_t stream,
                             const int max_threads_per_block) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int max_sequence_length = parameters.max_sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  const int H = head_size / 4;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(data.present_key),
                                                        reinterpret_cast<const float2*>(new_key),
                                                        data.seqlens_k_total,
                                                        past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(data.present_value),
                                                        reinterpret_cast<const float2*>(new_value),
                                                        data.seqlens_k_total,
                                                        past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  } else {
    int steps = int(ceil(float(H * kv_num_heads) / 256.0));
    const dim3 grid(steps, sequence_length, batch_size);
    const dim3 block(256, 1, 1);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(data.present_key),
                                                             reinterpret_cast<const float2*>(new_key),
                                                             data.seqlens_k_total,
                                                             past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(data.present_value),
                                                             reinterpret_cast<const float2*>(new_value),
                                                             data.seqlens_k_total,
                                                             past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  }
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<T>& data,
    CudaTuningContext* tuning_ctx) {
  cudaStream_t stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  // const int present_sequence_length = parameters.max_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  const void* query;
  const void* key;
  const void* value;
  if (!parameters.is_packed_qkv) {
    query = reinterpret_cast<const void*>(data.query);
    key = reinterpret_cast<const void*>(data.key);
    value = reinterpret_cast<const void*>(data.value);
  } else {
    size_t q_size = static_cast<size_t>(batch_size * sequence_length * num_heads * head_size);
    size_t k_size = static_cast<size_t>(batch_size * sequence_length * kv_num_heads * head_size);
    auto q = reinterpret_cast<T*>(data.unpacked_qkv_buffer);
    auto k = reinterpret_cast<T*>(data.unpacked_qkv_buffer + q_size);
    auto v = reinterpret_cast<T*>(data.unpacked_qkv_buffer + q_size + k_size);
    ORT_RETURN_IF_ERROR(LaunchUnpackQKV(reinterpret_cast<const T*>(data.query), q, k, v, num_heads, kv_num_heads,
                                        head_size, sequence_length, batch_size, stream, max_threads_per_block));
    query = reinterpret_cast<const void*>(q);
    key = reinterpret_cast<const void*>(k);
    value = reinterpret_cast<const void*>(v);
  }

  if (parameters.do_rotary) {
    size_t bsh = static_cast<size_t>(parameters.batch_size * parameters.sequence_length * parameters.head_size);
    size_t q_size = bsh * static_cast<size_t>(parameters.num_heads);
    size_t k_size = bsh * static_cast<size_t>(parameters.kv_num_heads);
    auto q_buffer = reinterpret_cast<T*>(data.rotary_buffer);
    auto k_buffer = q_buffer + q_size;
    auto position_ids_buff = reinterpret_cast<int64_t*>(k_buffer + k_size);
    ORT_RETURN_IF_ERROR(FillPositionIds(parameters, data.seqlens_k_total, position_ids_buff, stream,
                                        max_threads_per_block));
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("position_ids", position_ids_buff, batch_size, sequence_length);

    // Launch rotary embedding kernel. This requires separated Q, K and V
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(stream, q_buffer, reinterpret_cast<const T*>(query),
                                                       position_ids_buff, data.cos_cache, data.sin_cache,
                                                       parameters.batch_size, parameters.sequence_length,
                                                       parameters.num_heads, parameters.head_size,
                                                       parameters.rotary_dim, parameters.max_sequence_length,
                                                       /*position_ids_format*/ 1, parameters.rotary_interleaved,
                                                       device_prop.maxThreadsPerBlock, /*transposed*/ false));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(stream, k_buffer, reinterpret_cast<const T*>(key),
                                                       position_ids_buff, data.cos_cache, data.sin_cache,
                                                       parameters.batch_size, parameters.sequence_length,
                                                       parameters.kv_num_heads, parameters.head_size,
                                                       parameters.rotary_dim, parameters.max_sequence_length,
                                                       /*position_ids_format*/ 1, parameters.rotary_interleaved,
                                                       device_prop.maxThreadsPerBlock, /*transposed*/ false));
    query = reinterpret_cast<const void*>(q_buffer);
    key = reinterpret_cast<const void*>(k_buffer);
  }

  ORT_ENFORCE(parameters.past_present_share_buffer);
  ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(parameters, data, key, value, stream, max_threads_per_block));

  SparseAttentionTunableParams<T> params(
      tuning_ctx,
      ort_stream,
      data.output,
      data.query,
      data.key,
      data.value,
      parameters.batch_size,
      parameters.sequence_length,
      parameters.num_heads,
      parameters.kv_num_heads,
      parameters.head_size,
      parameters.total_sequence_length,
      parameters.scale,
      data.kernel_layout.block_size,                                      // kernel_block_size
      data.kernel_layout.csr_row_indices + data.kernel_layout.start_row,  // skip past_seq_len in row indices
      data.kernel_layout.csr_col_indices,                                 // (num_layout, num_rows, num_cols)
      data.kernel_layout.num_rows + 1,                                    // stride per head in row indices
      data.kernel_layout.num_rows * data.kernel_layout.num_cols,          // stride per head in col indices
      data.kernel_layout.num_layout);

  assert (tuning_ctx->IsTunableOpEnabled());
  static SparseAttentionTunableOp<T> op;
  return op(&params);
}

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<half>& data,
    CudaTuningContext* tuning_ctx);

template Status QkvToContext<BFloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<BFloat16>& data,
    CudaTuningContext* tuning_ctx);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
