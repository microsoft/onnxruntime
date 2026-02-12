/*
 The implementation of this file is based on our Multi-Head Attention impl.cu file,
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
// (1) support past state, unidirectional mask (causal)
// (2) use flash attention kernel from (https://github.com/Dao-AILab/flash-attention)
// (3) support different number of heads for Q and KV
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cub/cub.cuh>

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/utils/debug_macros.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/group_query_attention_qkv.cuh"
#include "contrib_ops/cuda/bert/group_query_attention_qdq.cuh"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "contrib_ops/cuda/bert/rotary_common.cuh"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_type_conversion.h"

#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;

using onnxruntime::contrib::GroupQueryAttentionParameters;
using onnxruntime::contrib::LAYOUT_BNSH;
using onnxruntime::contrib::cuda::GroupQueryAttentionData;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// ============================================================================
// QKV Preprocessing Helpers
// ============================================================================

// Internal helper to get Q, K, V pointers, handling packed input
//
// This function orchestrates the preparation of Q, K, and V tensors for attention kernels.
// It performs:
// 1. Handling packed vs. unpacked QKV inputs.
// 2. Managing KV cache updates (appending new tokens).
// 3. Ensuring synchronization between past and present KV caches when necessary.
// 4. Launching the UnpackRoPEQuantizeAppend kernel to unpack, apply RoPE, and update caches.
// 5. Returning strict Q, K, V pointers ready for the core attention operation.
template <typename T, typename U>
Status PrepareQKV(
    cudaStream_t stream,
    const int max_threads_per_block,
    const GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    const T*& q) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
  typedef typename onnxruntime::cuda::OrtToCudaType<U>::type CudaU;
  CudaT* q_out = reinterpret_cast<CudaT*>(data.qkv_buffer);

  if (!parameters.is_packed_qkv && !parameters.do_rotary) {
    q_out = nullptr;
  }

  CudaT* k = reinterpret_cast<CudaT*>(data.present_key);
  CudaT* v = reinterpret_cast<CudaT*>(data.present_value);
  int max_cache_length = parameters.seqlen_present_kv_cache;
  bool is_cache_bnsh = (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);

  if (!parameters.past_present_share_buffer) {
    size_t kv_buffer_size = (size_t)batch_size * kv_num_heads * max_cache_length * head_size * sizeof(CudaU);
    CUDA_CALL_THROW(cudaMemsetAsync(data.present_key, 0, kv_buffer_size, stream));
    CUDA_CALL_THROW(cudaMemsetAsync(data.present_value, 0, kv_buffer_size, stream));
  }

  // Copy past KV to present KV if needed
  if (!parameters.past_present_share_buffer && data.past_key != nullptr && parameters.seqlen_past_kv_cache > 0) {
    if (is_cache_bnsh) {
      size_t src_pitch = (size_t)parameters.seqlen_past_kv_cache * head_size * sizeof(CudaU);
      size_t dst_pitch = (size_t)parameters.seqlen_present_kv_cache * head_size * sizeof(CudaU);
      size_t width = src_pitch;
      size_t height = (size_t)batch_size * kv_num_heads;

      CUDA_CALL_THROW(cudaMemcpy2DAsync(data.present_key, dst_pitch, data.past_key, src_pitch, width, height,
                                        cudaMemcpyDeviceToDevice, stream));
      CUDA_CALL_THROW(cudaMemcpy2DAsync(data.present_value, dst_pitch, data.past_value, src_pitch, width, height,
                                        cudaMemcpyDeviceToDevice, stream));
    } else {
      size_t src_pitch = (size_t)parameters.seqlen_past_kv_cache * kv_num_heads * head_size * sizeof(CudaU);
      size_t dst_pitch = (size_t)parameters.seqlen_present_kv_cache * kv_num_heads * head_size * sizeof(CudaU);
      size_t width = src_pitch;
      size_t height = (size_t)batch_size;

      CUDA_CALL_THROW(cudaMemcpy2DAsync(data.present_key, dst_pitch, data.past_key, src_pitch, width, height,
                                        cudaMemcpyDeviceToDevice, stream));
      CUDA_CALL_THROW(cudaMemcpy2DAsync(data.present_value, dst_pitch, data.past_value, src_pitch, width, height,
                                        cudaMemcpyDeviceToDevice, stream));
    }
  }

  ORT_RETURN_IF_ERROR(LaunchUnpackRoPEAppend<CudaT>(
      parameters.is_packed_qkv ? reinterpret_cast<const CudaT*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.value),
      q_out, k, v, data.k_scale, data.v_scale,
      num_heads, kv_num_heads, head_size, sequence_length, batch_size,
      max_cache_length, data.past_seq_lens,
      reinterpret_cast<const CudaT*>(data.cos_cache), reinterpret_cast<const CudaT*>(data.sin_cache),
      parameters.rotary_dim, data.position_ids, parameters.rotary_interleaved,
      is_cache_bnsh, parameters.k_quant_type, parameters.kv_cache_bit_width,
      stream, max_threads_per_block));

  if (q_out != nullptr) {
    q = reinterpret_cast<const T*>(q_out);
  } else {
    q = reinterpret_cast<const T*>(data.query);
  }

  return Status::OK();
}

////////// Auxiliary Kernels for KV prep

// Concat new to past in present. Supports past BSNH or past BNSH
template <typename T, typename U>
Status LaunchConcatNewToPastKVHelper(GroupQueryAttentionParameters& parameters,
                                     GroupQueryAttentionData<T, U>& data,
                                     const void* new_key,
                                     const void* new_value,
                                     cudaStream_t stream,
                                     const int max_threads_per_block,
                                     const bool past_only = false,
                                     const T* cos_cache = nullptr,
                                     const T* sin_cache = nullptr,
                                     const int rotary_dim = 0,
                                     const int64_t* position_ids = nullptr,
                                     const bool interleaved = false) {
  const int batch_size = parameters.batch_size;
  const int kv_sequence_length = parameters.sequence_length;
  const int past_sequence_length = parameters.seqlen_past_kv_cache;
  const int present_sequence_length = parameters.seqlen_present_kv_cache;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  const bool is_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;

  return LaunchConcatNewToPastKV(batch_size,
                                 kv_num_heads,
                                 head_size,
                                 kv_sequence_length,
                                 past_sequence_length,
                                 present_sequence_length,
                                 is_bsnh,
                                 data.past_seq_lens,
                                 data.total_seq_lens,
                                 reinterpret_cast<const T*>(data.past_key),
                                 reinterpret_cast<const T*>(data.past_value),
                                 reinterpret_cast<const T*>(new_key),
                                 reinterpret_cast<const T*>(new_value),
                                 reinterpret_cast<T*>(data.present_key),
                                 reinterpret_cast<T*>(data.present_value),
                                 stream,
                                 max_threads_per_block,
                                 past_only,
                                 cos_cache,
                                 sin_cache,
                                 rotary_dim,
                                 position_ids,
                                 interleaved);
}

// Concat new to kv buffer in place
template <typename T, typename U>
Status LaunchConcatKVInPlace(GroupQueryAttentionParameters& parameters,
                             GroupQueryAttentionData<T, U>& data,
                             const void* new_key,
                             const void* new_value,
                             bool is_new_kv_bnsh_format,
                             cudaStream_t stream,
                             const int max_threads_per_block) {
  const int max_sequence_length = parameters.seqlen_present_kv_cache;

  assert(parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH ||
         parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  bool is_past_kv_bnsh_format = (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);

  return LaunchConcatKVInPlace(parameters.batch_size,
                               parameters.kv_num_heads,
                               parameters.head_size,
                               max_sequence_length,
                               data.past_seq_lens,
                               data.total_seq_lens,
                               parameters.sequence_length,
                               reinterpret_cast<const T*>(new_key),
                               reinterpret_cast<const T*>(new_value),
                               reinterpret_cast<T*>(data.present_key),
                               reinterpret_cast<T*>(data.present_value),
                               is_past_kv_bnsh_format,
                               is_new_kv_bnsh_format,
                               stream,
                               max_threads_per_block);
}

// ============================================================================
// Ungroup Kernel
// ============================================================================
// PURPOSE:
//   Expands grouped KV heads to match Q heads for Memory Efficient Attention.
//   Each KV head is replicated q_num_heads/kv_num_heads times.
//
// INPUTS:
//   kv_in      - Grouped KV tensor with kv_num_heads heads
//   in_seqlen  - Sequence length of input tensor
//   kv_num_heads - Number of KV heads (fewer than Q heads)
//   is_bsnh    - True for BSNH format, False for BNSH format
//
// OUTPUTS:
//   kv_out     - Ungrouped tensor with q_num_heads heads (BSNH format)
//
// THREAD MAPPING:
//   threadIdx.x = h (head dimension element)
//   threadIdx.y = out_n (output head index)
//   blockIdx.x  = s (sequence position)
//   blockIdx.y  = b (batch index)
//
// ASSUMPTIONS:
//   - q_num_heads is divisible by kv_num_heads
//   - H * q_num_heads <= max_threads_per_block (use UngroupLarge otherwise)
// ============================================================================
template <typename T>
__global__ void Ungroup(const T* kv_in,
                        T* kv_out,
                        const int in_seqlen,
                        const int kv_num_heads,
                        const bool is_bsnh) {
  const int h = threadIdx.x;
  const int out_n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int out_seqlen = gridDim.x;
  const int q_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int q_kv_head_ratio = q_num_heads / kv_num_heads;
  const int out_batch_stride = out_seqlen * q_num_heads * H;
  const int out_row_stride = is_bsnh ? q_num_heads * H : H;
  const int out_head_stride = is_bsnh ? H : out_seqlen * H;

  const int in_batch_stride = in_seqlen * kv_num_heads * H;
  const int in_row_stride = is_bsnh ? kv_num_heads * H : H;
  const int in_head_stride = is_bsnh ? H : in_seqlen * H;
  const int in_n = out_n / q_kv_head_ratio;

  const int out_offset = out_batch_stride * b + out_row_stride * s + out_head_stride * out_n + h;
  const int in_offset = in_batch_stride * b + in_row_stride * s + in_head_stride * in_n + h;
  kv_out[out_offset] = kv_in[in_offset];
}

// ============================================================================
// UngroupLarge Kernel
// ============================================================================
// PURPOSE:
//   Same as Ungroup but for cases where H * q_num_heads > max_threads_per_block.
//   Uses a 1D thread grid to avoid block dimension limit.
//
// THREAD MAPPING:
//   Each thread processes one element indexed by (threadIdx.x + blockDim.x * blockIdx.x)
//   This linear index is decomposed into (h, out_n) within the kernel.
//   blockIdx.y = s (sequence position)
//   blockIdx.z = b (batch index)
// ============================================================================
template <typename T>
__global__ void UngroupLarge(const T* kv_in,
                             T* kv_out,
                             const int H,
                             const int in_seqlen,
                             const int q_num_heads,
                             const int kv_num_heads,
                             const bool is_bsnh) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);  // index along H * q_num_heads elements
  if (i < H * q_num_heads) {
    const int out_seqlen = gridDim.y;
    const int s = blockIdx.y;
    const int b = blockIdx.z;

    const int q_kv_head_ratio = q_num_heads / kv_num_heads;
    const int out_batch_stride = out_seqlen * q_num_heads * H;
    const int out_row_stride = is_bsnh ? q_num_heads * H : H;
    const int out_head_stride = is_bsnh ? H : out_seqlen * H;

    const int in_batch_stride = in_seqlen * kv_num_heads * H;
    const int in_row_stride = is_bsnh ? kv_num_heads * H : H;
    const int in_head_stride = is_bsnh ? H : in_seqlen * H;

    const int h = i % H;
    const int out_n = i / H;
    const int in_n = out_n / q_kv_head_ratio;
    const int out_offset = out_batch_stride * b + out_row_stride * s + out_head_stride * out_n + h;
    const int in_offset = in_batch_stride * b + in_row_stride * s + in_head_stride * in_n + h;
    kv_out[out_offset] = kv_in[in_offset];
  }
}

// Ungroup kv or present kv for use in Memory Efficient kernel. If present kv is not null and is BNSH, transposes it.
template <typename T>
Status LaunchUngroup(const GroupQueryAttentionParameters& parameters,
                     float2* k_buff, float2* v_buff,
                     const float2* k_og, const float2* v_og,
                     const int buff_seqlen, const int og_seqlen,
                     const bool is_bsnh,
                     cudaStream_t stream,
                     const int max_threads_per_block) {
  const int batch_size = parameters.batch_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  const int H = head_size / 4;
  if (H * num_heads <= max_threads_per_block) {
    const dim3 grid(buff_seqlen, batch_size, 1);
    const dim3 block(H, num_heads, 1);
    Ungroup<float2><<<grid, block, 0, stream>>>(k_og,
                                                k_buff,
                                                og_seqlen,
                                                kv_num_heads,
                                                is_bsnh);
    Ungroup<float2><<<grid, block, 0, stream>>>(v_og,
                                                v_buff,
                                                og_seqlen,
                                                kv_num_heads,
                                                is_bsnh);
  } else {
    int steps = int(ceil(float(H * num_heads) / 256.0));
    const dim3 grid(steps, buff_seqlen, batch_size);
    const dim3 block(256, 1, 1);
    UngroupLarge<float2><<<grid, block, 0, stream>>>(k_og,
                                                     k_buff,
                                                     H,
                                                     og_seqlen,
                                                     num_heads,
                                                     kv_num_heads,
                                                     is_bsnh);
    UngroupLarge<float2><<<grid, block, 0, stream>>>(v_og,
                                                     v_buff,
                                                     H,
                                                     og_seqlen,
                                                     num_heads,
                                                     kv_num_heads,
                                                     is_bsnh);
  }
  return CUDA_CALL(cudaGetLastError());
}

// ============================================================================
// UnpackQKV Kernel
// ============================================================================
// PURPOSE:
//   Unpacks packed QKV tensor into separate Q, K, V tensors.
//   Packed input has interleaved [Q, K, V] per token.
//
// INPUTS:
//   packed_qkv - Input tensor of shape [B, S, (Q_heads + 2*KV_heads) * head_size]
//   num_heads  - Number of Q heads
//   kv_num_heads - Number of KV heads
//   head_size  - Head dimension
//   sequence_length - Token sequence length
//   batch_size - Batch size
//
// OUTPUTS:
//   unpacked_q - Q tensor [B, S, num_heads, head_size] if BSNH, or [B, num_heads, S, head_size] if BNSH
//   unpacked_k - K tensor [B, S, kv_num_heads, head_size] if BSNH, or [B, kv_num_heads, S, head_size] if BNSH
//   unpacked_v - V tensor (same layout as K)
//
// TEMPLATE PARAM:
//   output_bnsh - If true, outputs BNSH format; if false, outputs BSNH format
//
// THREAD MAPPING:
//   One thread per element in packed_qkv. Thread determines which of Q/K/V
//   the element belongs to based on the offset within the hidden dimension.
// ============================================================================
template <typename T, bool output_bnsh>
__global__ void UnpackQKV(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
                          const int kv_num_heads, const int head_size, const int sequence_length,
                          const int batch_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int d = (num_heads + 2 * kv_num_heads) * head_size;
  const int qkv_size = batch_size * sequence_length * d;
  const int q_hidden = num_heads * head_size;
  const int k_hidden = kv_num_heads * head_size;
  if (tid < qkv_size) {
    int b = tid / (d * sequence_length);
    int s = (tid % (d * sequence_length)) / d;
    int offset = tid % d;
    if (output_bnsh) {  // output BNSH
      int head_count = kv_num_heads;
      T* unpacked = nullptr;
      if (offset < q_hidden) {
        unpacked = unpacked_q;
        head_count = num_heads;
      } else if (offset < q_hidden + k_hidden) {
        unpacked = unpacked_k;
        offset -= q_hidden;
      } else {
        unpacked = unpacked_v;
        offset -= (q_hidden + k_hidden);
      }

      if (unpacked != nullptr) {
        int n = offset / head_size;
        int h = offset % head_size;

        int unpacked_i = INDEX_4D(head_count, sequence_length, head_size, b, n, s, h);
        unpacked[unpacked_i] = packed_qkv[tid];
      } else {
#ifndef NDEBUG
        assert(false && "Unexpected null 'unpacked' pointer in GroupQueryAttention unpack kernel");
#endif
      }
    } else {  // output BSNH
      if (offset < q_hidden) {
        if (unpacked_q != nullptr) {
          int unpacked_i = b * sequence_length * num_heads * head_size + s * num_heads * head_size + offset;
          unpacked_q[unpacked_i] = packed_qkv[tid];
        }
      } else if (offset < q_hidden + k_hidden) {
        if (unpacked_k != nullptr) {
          int unpacked_i = b * sequence_length * kv_num_heads * head_size +
                           s * kv_num_heads * head_size + (offset - q_hidden);
          unpacked_k[unpacked_i] = packed_qkv[tid];
        }
      } else {
        if (unpacked_v != nullptr) {
          int unpacked_i = b * sequence_length * kv_num_heads * head_size +
                           s * kv_num_heads * head_size + (offset - q_hidden - k_hidden);
          unpacked_v[unpacked_i] = packed_qkv[tid];
        }
      }
    }
  }
}

// Unpack packed qkv
template <typename T, bool output_bnsh>
Status LaunchUnpackQKV(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
                       const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size,
                       cudaStream_t stream, const int max_threads_per_block) {
  const int threads = max_threads_per_block;
  const int blocks = (batch_size * sequence_length * (num_heads + 2 * kv_num_heads) * head_size + threads - 1) / threads;
  UnpackQKV<T, output_bnsh><<<blocks, threads, 0, stream>>>(
      packed_qkv, unpacked_q, unpacked_k, unpacked_v, num_heads, kv_num_heads, head_size, sequence_length, batch_size);
  return CUDA_CALL(cudaGetLastError());
}

// ============================================================================
// GetSequenceLengths Kernel
// ============================================================================
// PURPOSE:
//   Computes derived sequence length buffers from input seqlens_k.
//   Input seqlens_k contains (total_sequence_length - 1) for historical reasons.
//
// INPUTS:
//   total_seq_lens_minus_one - Input from ONNX graph: total_len - 1 per batch [B]
//   sequence_length          - Current Q sequence length (new tokens)
//   is_first_prompt          - True if this is the first prompt (no past)
//
// OUTPUTS:
//   past_seq_lens   - Offset where new KV should be appended [B]
//                     First prompt: 0
//                     Otherwise: total_len - sequence_length
//   total_seq_lens  - Total valid tokens including new ones [B]
//   padded_seq_lens - Padded length for masking (first prompt only) [B]
//                     First prompt: sequence_length
//                     Otherwise: not set (undefined)
//
// THREAD MAPPING:
//   One thread per batch element.
//
// USAGE:
//   Called once per inference to derive all sequence length variants.
// ============================================================================
__global__ void GetSequenceLengths(const int* total_seq_lens_minus_one,
                                   int* past_seq_lens,
                                   int* total_seq_lens,
                                   int* padded_seq_lens,
                                   const int batch_size,
                                   const int sequence_length,
                                   const bool is_first_prompt) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < batch_size) {
    const int total_len = total_seq_lens_minus_one[i] + 1;
    total_seq_lens[i] = total_len;
    if (is_first_prompt) {
      past_seq_lens[i] = 0;
      padded_seq_lens[i] = sequence_length;
    } else {
      past_seq_lens[i] = total_len - sequence_length;
      padded_seq_lens[i] = 0;
    }
  }
}

Status LaunchGetSequenceLengths(
    const int* total_seq_lens_minus_one,
    int* past_seq_lens,
    int* total_seq_lens,
    int* padded_seq_lens,
    const int batch_size,
    const int sequence_length,
    const bool is_first_prompt,
    cudaStream_t stream,
    const int max_threads_per_block) {
  int blocks = (batch_size + max_threads_per_block - 1) / max_threads_per_block;
  GetSequenceLengths<<<blocks, max_threads_per_block, 0, stream>>>(total_seq_lens_minus_one, past_seq_lens, total_seq_lens, padded_seq_lens, batch_size, sequence_length, is_first_prompt);
  return CUDA_CALL(cudaGetLastError());
}

// Trace function for debugging
#define ORT_GQA_TRACE(func_name)                                                                                          \
  DUMP_PRINTF("[GQA %s] is_packed_qkv: %d, is_first_prompt: %d, is_subsequent_prompt: %d, past_present_share_buffer: %d", \
              func_name,                                                                                                  \
              static_cast<int>(parameters.is_packed_qkv),                                                                 \
              static_cast<int>(parameters.is_first_prompt),                                                               \
              static_cast<int>(parameters.is_subsequent_prompt),                                                          \
              static_cast<int>(parameters.past_present_share_buffer));

////////// Kernels (supports right padding but not left padding)
// Use flash attention for all workloads (rotary, kv append, attention, etc.). No extra kernel is used in this path.
// Currently, only decoding or subsequent prompt can use this path. First prompt will not use this path.
template <typename T, typename U>
Status ExtremeDecoding(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  ORT_GQA_TRACE("ExtremeDecoding");

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  // const int kv_sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  // bool is_causal = parameters.is_unidirectional;
  // bool is_bf16 = std::is_same<T, BFloat16>::value;

  typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
  bool past_bsnh = (past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);

  // Ultimate Fused Preprocessing: Unpack, RoPE Q, RoPE K, Quantize K/V, Append K/V
  // This replaces all manual steps (Rotate Q, Rotate K, Quantize, StridedCopy)
  CudaT* q_rot_ptr = reinterpret_cast<CudaT*>(data.qkv_buffer);
  const CudaT* q_input_for_xqa = q_rot_ptr;
  if (q_rot_ptr == nullptr) {
    q_input_for_xqa = reinterpret_cast<const CudaT*>(data.query);
  }

  ORT_RETURN_IF_ERROR(LaunchUnpackRoPEAppend<CudaT>(
      parameters.is_packed_qkv ? reinterpret_cast<const CudaT*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.value),
      q_rot_ptr,  // unpacked_q (can be null if !do_rotary)
      data.present_key,
      data.present_value,
      data.k_scale,
      data.v_scale,
      num_heads,
      kv_num_heads,
      head_size,
      sequence_length,
      batch_size,
      parameters.seqlen_present_kv_cache,  // max_seqlen (capacity)
      data.past_seq_lens,
      reinterpret_cast<const CudaT*>(data.cos_cache),
      reinterpret_cast<const CudaT*>(data.sin_cache),
      parameters.do_rotary ? parameters.rotary_dim : 0,
      data.position_ids,
      parameters.rotary_interleaved,
      !past_bsnh,  // is_cache_bnsh
      parameters.k_quant_type,
      parameters.kv_cache_bit_width,
      stream,
      device_prop.maxThreadsPerBlock));

  // Determine workspace size for XQA
  void* xqa_workspace = data.xqa_buffer;
  size_t xqa_workspace_size = data.xqa_buffer_bytes;

  // 5. Launch XQA
  Status status = onnxruntime::contrib::cuda::LaunchXQAKernel<CudaT>(
      device_prop,
      stream,
      q_input_for_xqa,
      data.present_key,
      data.present_value,
      data.output,
      batch_size,
      num_heads,
      kv_num_heads,
      parameters.head_size,
      parameters.seqlen_present_kv_cache,  // max_seq_len (Capacity)
      scale,
      past_bsnh,
      data.past_seq_lens,
      data.k_scale,  // kv_cache_scale
      // Map KVQuantizationType (0=NONE, 1=TENSOR, 2=CHANNEL) to XqaQuantType (0=FP16/BF16, 1=INT8, 2=FP8)
      (parameters.k_quant_type == KVQuantizationType::NONE) ? onnxruntime::contrib::cuda::XqaQuantType::kNone : onnxruntime::contrib::cuda::XqaQuantType::kInt8,
      xqa_workspace,
      xqa_workspace_size);

  // If XQA launch fails, debugging info

  return status;
}

////////// Kernels (supports right padding but not left padding)
// Use flash attention for all workloads (rotary, kv append, attention, etc.). No extra kernel is used in this path.
// Currently, only decoding or subsequent prompt can use this path. First prompt will not use this path.
#if USE_FLASH_ATTENTION

// Use flash attention for all workloads (rotary, kv append, attention, etc.). No extra kernel is used in this path.
// Currently, only decoding or subsequent prompt can use this path. First prompt will not use this path.
template <typename T, typename U>
Status FlashDecoding(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  assert(!parameters.is_first_prompt && parameters.past_present_share_buffer);

  ORT_GQA_TRACE("FlashDecoding");

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value || std::is_same<T, BFloat16>::value;

  void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
  void* key;
  void* value;

  if (!parameters.is_packed_qkv) {
    key = reinterpret_cast<void*>(const_cast<T*>(data.key));
    value = reinterpret_cast<void*>(const_cast<T*>(data.value));
  } else {
    const size_t key_offset = static_cast<size_t>(num_heads * head_size);
    const size_t value_offset = static_cast<size_t>(kv_num_heads * head_size);
    key = reinterpret_cast<T*>(query) + key_offset;
    value = reinterpret_cast<T*>(key) + value_offset;
  }

  void* seqlens_k = reinterpret_cast<void*>(data.past_seq_lens);

  void* present_key = data.present_key;
  void* present_value = data.present_value;
  void* cos_cache = reinterpret_cast<void*>(const_cast<T*>(data.cos_cache));
  void* sin_cache = reinterpret_cast<void*>(const_cast<T*>(data.sin_cache));
  void* head_sink = reinterpret_cast<void*>(const_cast<T*>(data.head_sink));

  bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;

  DUMP_PRINTF("[FlashDecoding] key=%p, value=%p, present_key=%p, present_value=%p, seqlens_k=%p, is_packed_qkv=%d",
              key, value, present_key, present_value, seqlens_k, static_cast<int>(parameters.is_packed_qkv));

  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
      device_prop, stream, query, present_key, present_value, key, value, data.output,
      reinterpret_cast<void*>(data.softmax_lse), seqlens_k, cos_cache, sin_cache,
      /*cache_batch_idx*/ nullptr, /*leftpad_k*/ nullptr, head_sink, /*block_table*/ nullptr,
      batch_size, num_heads, kv_num_heads, head_size, sequence_length,
      parameters.seqlen_present_kv_cache, kv_sequence_length, parameters.rotary_dim,
      scale, parameters.softcap, is_causal, is_bf16, parameters.use_smooth_softmax, past_bsnh, parameters.num_splits,
      reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum),
      parameters.local_window_size - 1, parameters.rotary_interleaved, parameters.is_packed_qkv,
      0, 1));

  return Status::OK();
}

// Use extra kernel(s) for unpacking, rotary and kv append.
// Flash attention is used for attention only.
template <typename T, typename U>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value || std::is_same<T, BFloat16>::value;

  DUMP_TENSOR_INIT();

  const T* q_prep = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQKV<T, U>(stream, max_threads_per_block, parameters, data, q_prep)));

  void* query = const_cast<T*>(q_prep);
  void* present_key = data.present_key;
  void* present_value = data.present_value;

  // Disable internal RoPE in Flash Attention (pass nullptr)
  void* cos_cache = nullptr;
  void* sin_cache = nullptr;
  void* head_sink = reinterpret_cast<void*>(const_cast<T*>(data.head_sink));

  // We have already appended (and quantized if needed) the new tokens into present_key/value.
  // Pass nullptr for new_k/new_v to disable flash attention kernel's internal Append_KV logic.
  void* kernel_new_k = nullptr;
  void* kernel_new_v = nullptr;

  // Use padded seq lens for first prompt since mha_fwd_kvcache assumes uniform seqlen_q.
  int* seq_lens = parameters.is_first_prompt ? data.padded_seq_lens : data.total_seq_lens;

  // After PrepareQKV, the input for flash attention is no longer packed.
  constexpr bool is_packed_qkv_for_flash = false;

  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
      device_prop, stream, query, present_key, present_value,
      kernel_new_k, kernel_new_v,
      data.output, reinterpret_cast<void*>(data.softmax_lse), seq_lens, cos_cache, sin_cache,
      /*cache_batch_idx*/ nullptr, /*leftpad_k*/ nullptr, head_sink, /*block_table*/ nullptr,
      batch_size, num_heads, kv_num_heads, head_size, sequence_length,
      parameters.seqlen_present_kv_cache, kv_sequence_length,
      0,  // rotary_dim = 0 as it is already rotated
      scale, parameters.softcap, is_causal, is_bf16,
      parameters.use_smooth_softmax, past_bsnh, parameters.num_splits,
      reinterpret_cast<void*>(data.softmax_lse_accum),
      reinterpret_cast<void*>(data.out_accum), parameters.local_window_size - 1,
      parameters.rotary_interleaved, is_packed_qkv_for_flash, 0, 1));

  DUMP_TENSOR("Total Seq Lens", data.total_seq_lens, batch_size, 1);
  DUMP_TENSOR("Past Seq Lens", data.past_seq_lens, batch_size, 1);

  return Status::OK();
}

// Fallback path for decoding quantized kv cache, when XQA is not usable (due to softcap, window, etc.)
// We dequantize the cache and run standard Flash Attention.
template <typename T, typename U>
Status DequantizeFlashAttentionFallback(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  assert(!parameters.is_first_prompt);  // Only support first prompt for this function.
  assert(parameters.k_quant_type != KVQuantizationType::NONE || parameters.v_quant_type != KVQuantizationType::NONE);

  ORT_GQA_TRACE("DequantizeFlashAttentionFallback");

  // We need to dequantize the entire KV cache (present_key/value) into a float/half buffer (data.qkv_buffer).
  // Layout in qkv_buffer: [Q (rotated)] [K_dequantized] [V_dequantized]
  typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
  CudaT* q_rot = reinterpret_cast<CudaT*>(data.qkv_buffer);
  size_t q_elements = static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads * parameters.head_size;
  size_t k_elements = static_cast<size_t>(parameters.batch_size) * parameters.seqlen_present_kv_cache * parameters.kv_num_heads * parameters.head_size;
  CudaT* k_dequant = q_rot + q_elements;
  CudaT* v_dequant = k_dequant + k_elements;

  // Step 1: Update Quantized Cache
  // We can use LaunchUnpackRoPEQuantizeAppend to unpack new QKV, apply RoPE, and append to quantized cache.
  // This will also put rotated Q into q_rot.
  ORT_RETURN_IF_ERROR(LaunchUnpackRoPEAppend<CudaT>(
      parameters.is_packed_qkv ? reinterpret_cast<const CudaT*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.value),
      q_rot, data.present_key, data.present_value, data.k_scale, data.v_scale,
      parameters.num_heads, parameters.kv_num_heads, parameters.head_size, parameters.sequence_length, parameters.batch_size,
      parameters.seqlen_present_kv_cache, data.past_seq_lens,
      reinterpret_cast<const CudaT*>(data.cos_cache), reinterpret_cast<const CudaT*>(data.sin_cache),
      parameters.rotary_dim, data.position_ids, parameters.rotary_interleaved,
      (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH),
      parameters.k_quant_type, parameters.kv_cache_bit_width,
      stream, device_prop.maxThreadsPerBlock));

  // Step 2: Dequantize Entire Cache
  // We now have the updated quantized cache in data.present_key/value. We need to dequantize it to k_dequant/v_dequant.
  bool is_bsnh = (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);

  if (parameters.kv_cache_bit_width == 8) {
    ORT_RETURN_IF_ERROR((LaunchDequantizeKV<CudaT, int8_t, float>(
        stream, k_dequant, reinterpret_cast<const int8_t*>(data.present_key), data.k_scale,
        nullptr, parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache,
        parameters.head_size, 8, parameters.k_quant_type, is_bsnh)));

    ORT_RETURN_IF_ERROR((LaunchDequantizeKV<CudaT, int8_t, float>(
        stream, v_dequant, reinterpret_cast<const int8_t*>(data.present_value), data.v_scale,
        nullptr, parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache,
        parameters.head_size, 8, parameters.v_quant_type, is_bsnh)));
#ifdef USE_INT4_KV_CACHE
  } else if (parameters.kv_cache_bit_width == 4) {
    // Int4 support if needed
    ORT_RETURN_IF_ERROR((LaunchDequantizeKV<CudaT, uint8_t, float>(
        stream, k_dequant, reinterpret_cast<const uint8_t*>(data.present_key), data.k_scale,
        nullptr, parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache,
        parameters.head_size, 4, parameters.k_quant_type, is_bsnh)));

    ORT_RETURN_IF_ERROR((LaunchDequantizeKV<CudaT, uint8_t, float>(
        stream, v_dequant, reinterpret_cast<const uint8_t*>(data.present_value), data.v_scale,
        nullptr, parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache,
        parameters.head_size, 4, parameters.v_quant_type, is_bsnh)));
#endif
  }

  // Step 3: Run Flash Attention on dequantized k/v
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value || std::is_same<T, BFloat16>::value;

  // Use the total_seq_lens here since k_dequant/v_dequant has both past and new tokens.
  void* seqlens_k_ptr = const_cast<void*>(reinterpret_cast<const void*>(data.total_seq_lens));
  int local_window_size = parameters.local_window_size > 0 ? parameters.local_window_size - 1 : -1;

  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
      device_prop, stream, q_rot, k_dequant, v_dequant, nullptr /*new K*/, nullptr /*new V*/, data.output,
      reinterpret_cast<void*>(data.softmax_lse), seqlens_k_ptr, nullptr /*cos_cache*/, nullptr /*sin_cache*/,
      /*cache_batch_idx*/ nullptr, /*leftpad_k*/ nullptr, reinterpret_cast<void*>(const_cast<T*>(data.head_sink)), /*block_table*/ nullptr,
      parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, parameters.head_size, parameters.sequence_length,
      parameters.seqlen_present_kv_cache, parameters.sequence_length, 0 /*rotary_dim = 0 as it is already rotated*/,
      scale, parameters.softcap, is_causal, is_bf16, parameters.use_smooth_softmax, is_bsnh, parameters.num_splits,
      reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum),
      local_window_size, parameters.rotary_interleaved, false,
      0, 1));

  return Status::OK();
}

// Use Flash Attention for float key and value, then quantize key/value to int8 to save to k/v cache.
template <typename T, typename U>
Status FlashAttentionAndQuantizeKV(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  assert(parameters.is_first_prompt);  // Only support first prompt for this function.
  assert(parameters.k_quant_type != KVQuantizationType::NONE || parameters.v_quant_type != KVQuantizationType::NONE);

  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int num_heads = parameters.num_heads;
  const int head_size = parameters.head_size;

  ORT_GQA_TRACE("FlashAttentionAndQuantizeKV");

  bool past_bsnh = parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;

  size_t q_elements = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size;
  size_t k_elements = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size;

  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  CudaT* q_final = reinterpret_cast<CudaT*>(data.qkv_buffer);

  // For FlashAttentionAndQuantizeKV, we need float K and V for attention.
  // We'll write them to qkv_buffer.
  CudaT* k_final = q_final + q_elements;
  CudaT* v_final = k_final + k_elements;

  ORT_RETURN_IF_ERROR(LaunchUnpackRoPEAppend<CudaT>(
      parameters.is_packed_qkv ? reinterpret_cast<const CudaT*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const CudaT*>(data.value),
      q_final, k_final, v_final, nullptr, nullptr,
      num_heads, kv_num_heads, head_size, sequence_length, batch_size,
      sequence_length, data.past_seq_lens,
      reinterpret_cast<const CudaT*>(data.cos_cache), reinterpret_cast<const CudaT*>(data.sin_cache),
      parameters.rotary_dim, data.position_ids, parameters.rotary_interleaved,
      false,  // BSNH for scratch
      KVQuantizationType::NONE,
      0,  // bit_width is 0 since we are not quantizing here.
      stream, max_threads_per_block));

  // 2. Run Float Flash Attention
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value || std::is_same<T, BFloat16>::value;

  int local_window_size = parameters.local_window_size > 0 ? parameters.local_window_size - 1 : -1;

  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
      device_prop, stream, q_final, k_final, v_final, data.output,
      reinterpret_cast<void*>(data.softmax_lse),
      batch_size, num_heads, kv_num_heads, head_size, sequence_length, sequence_length,
      scale, parameters.softcap, is_causal, is_bf16, parameters.use_smooth_softmax,
      parameters.num_splits,
      reinterpret_cast<void*>(data.softmax_lse_accum),
      reinterpret_cast<void*>(data.out_accum),
      true,  // kv_bsnh = true (BSNH)
      local_window_size));

  // 3. Quantize K and V to present cache
  if (parameters.k_quant_type != KVQuantizationType::NONE) {
    if (parameters.kv_cache_bit_width == 8) {
      ORT_RETURN_IF_ERROR((LaunchQuantizeKV<CudaT, int8_t, float>(
          stream, reinterpret_cast<int8_t*>(data.present_key), reinterpret_cast<const CudaT*>(k_final), data.k_scale,
          nullptr, data.total_seq_lens, batch_size, kv_num_heads, sequence_length, parameters.seqlen_present_kv_cache,
          head_size, 8, parameters.k_quant_type, true, past_bsnh)));
#ifdef USE_INT4_KV_CACHE
    } else if (parameters.kv_cache_bit_width == 4) {
      ORT_RETURN_IF_ERROR((LaunchQuantizeKV<CudaT, uint8_t, float>(
          stream, reinterpret_cast<uint8_t*>(data.present_key), reinterpret_cast<const CudaT*>(k_final), data.k_scale,
          nullptr, data.total_seq_lens, batch_size, kv_num_heads, sequence_length, parameters.seqlen_present_kv_cache,
          head_size, 4, parameters.k_quant_type, true, past_bsnh)));
#endif
    }
  }

  if (parameters.v_quant_type != KVQuantizationType::NONE) {
    if (parameters.kv_cache_bit_width == 8) {
      ORT_RETURN_IF_ERROR((LaunchQuantizeKV<CudaT, int8_t, float>(
          stream, reinterpret_cast<int8_t*>(data.present_value), reinterpret_cast<const CudaT*>(v_final), data.v_scale,
          nullptr, data.total_seq_lens, batch_size, kv_num_heads, sequence_length, parameters.seqlen_present_kv_cache,
          head_size, 8, parameters.v_quant_type, true, past_bsnh)));
#ifdef USE_INT4_KV_CACHE
    } else if (parameters.kv_cache_bit_width == 4) {
      ORT_RETURN_IF_ERROR((LaunchQuantizeKV<CudaT, uint8_t, float>(
          stream, reinterpret_cast<uint8_t*>(data.present_value), reinterpret_cast<const CudaT*>(v_final), data.v_scale,
          nullptr, data.total_seq_lens, batch_size, kv_num_heads, sequence_length, parameters.seqlen_present_kv_cache,
          head_size, 4, parameters.v_quant_type, true, past_bsnh)));
#endif
    }
  }

  return Status::OK();
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
template <typename T, typename U>
Status EfficientAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int present_sequence_length = parameters.seqlen_present_kv_cache;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  ORT_GQA_TRACE("EfficientAttention");

  const T* q_prep = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQKV<T, U>(stream, max_threads_per_block, parameters, data, q_prep)));

  const void* query = reinterpret_cast<const void*>(q_prep);
  const void* key;
  const void* value;
  const bool is_kv_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
  if (num_heads == kv_num_heads) {
    // Use present kv directly if not grouped
    key = reinterpret_cast<const void*>(data.present_key);
    value = reinterpret_cast<const void*>(data.present_value);
  } else {
    // Otherwise we use intermediate buffers to run memory efficient attention... best avoid this path
    float2* k_buff = reinterpret_cast<float2*>(data.k);
    float2* v_buff = reinterpret_cast<float2*>(data.v);
    const float2* k_og = reinterpret_cast<const float2*>(data.present_key);
    const float2* v_og = reinterpret_cast<const float2*>(data.present_value);

    ORT_RETURN_IF_ERROR(LaunchUngroup<T>(parameters, k_buff, v_buff, k_og, v_og, present_sequence_length,
                                         present_sequence_length, is_kv_bsnh, stream, max_threads_per_block));
    key = reinterpret_cast<const void*>(data.k);
    value = reinterpret_cast<const void*>(data.v);
  }

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_bf16 = std::is_same<T, __nv_bfloat16>::value || std::is_same<T, BFloat16>::value;
  p.is_half = !p.is_bf16 && (sizeof(T) == 2);
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.sequence_length = sequence_length;
  p.kv_sequence_length = present_sequence_length;  // maybe remove
  p.max_sequence_length = present_sequence_length;
  p.qk_head_size = head_size;
  p.v_head_size = head_size;
  p.causal = true;
  p.scale = scale;
  p.softcap = parameters.softcap;
  p.seqlen_k_ptr = parameters.is_first_prompt ? data.padded_seq_lens : data.total_seq_lens;
  p.seqstart_q_ptr = nullptr;
  p.seqstart_k_ptr = nullptr;
  p.query = query;
  p.key = key;
  p.value = value;
  p.attn_bias = nullptr;
  p.is_kv_bsnh = is_kv_bsnh;
  p.output = data.output;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(p.v_head_size, sizeof(T) == sizeof(float))
                    ? data.fmha_buffer
                    : nullptr;
  p.stream = stream;
  p.has_custom_right_padding = true;
  p.use_smooth_softmax = parameters.use_smooth_softmax;
  p.local_window_size = parameters.local_window_size;
  run_memory_efficient_attention(p);

  return Status::OK();
}
#endif

////////// API Functions

template <typename T, typename U>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& /*cublas*/,
    Stream* ort_stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;
  if (data.use_xqa) {
    return ExtremeDecoding(device_prop, stream, parameters, data, scale);
  }

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention_fast_decode) {
    return FlashDecoding(device_prop, stream, parameters, data, scale);
  }

  if (data.use_flash_attention) {
    if (parameters.k_quant_type != KVQuantizationType::NONE || parameters.v_quant_type != KVQuantizationType::NONE) {
      if (parameters.is_first_prompt) {
        return FlashAttentionAndQuantizeKV(device_prop, stream, parameters, data, scale);
      } else {
        return DequantizeFlashAttentionFallback(device_prop, stream, parameters, data, scale);
      }
    }

    return FlashAttention(device_prop, stream, parameters, data, scale);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    return EfficientAttention(device_prop, stream, parameters, data, scale);
  }
#endif

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unfused Group Query Attention not implemented yet.");
}

template struct GroupQueryAttentionData<half, half>;
template struct GroupQueryAttentionData<__nv_bfloat16, __nv_bfloat16>;
template struct GroupQueryAttentionData<BFloat16, BFloat16>;
template struct GroupQueryAttentionData<half, int8_t>;

template Status QkvToContext<half, half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<half, half>& data);

template Status QkvToContext<__nv_bfloat16, __nv_bfloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<__nv_bfloat16, __nv_bfloat16>& data);

template Status QkvToContext<BFloat16, BFloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<BFloat16, BFloat16>& data);

template Status QkvToContext<half, int8_t>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<half, int8_t>& data);

template struct GroupQueryAttentionData<__nv_bfloat16, int8_t>;

template Status QkvToContext<__nv_bfloat16, int8_t>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<__nv_bfloat16, int8_t>& data);

template struct GroupQueryAttentionData<half, uint8_t>;

template Status QkvToContext<half, uint8_t>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<half, uint8_t>& data);

template struct GroupQueryAttentionData<__nv_bfloat16, uint8_t>;

template Status QkvToContext<__nv_bfloat16, uint8_t>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<__nv_bfloat16, uint8_t>& data);

template Status LaunchUnpackQKV<half, LAYOUT_BNSH>(const half* packed_qkv, half* unpacked_q, half* unpacked_k, half* unpacked_v, const int num_heads, const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size, cudaStream_t stream, const int max_threads_per_block);
template Status LaunchUnpackQKV<__nv_bfloat16, LAYOUT_BNSH>(const __nv_bfloat16* packed_qkv, __nv_bfloat16* unpacked_q, __nv_bfloat16* unpacked_k, __nv_bfloat16* unpacked_v, const int num_heads, const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size, cudaStream_t stream, const int max_threads_per_block);

// BFloat16 variant is used in sparse attention.
template Status LaunchUnpackQKV<BFloat16, LAYOUT_BNSH>(const BFloat16* packed_qkv, BFloat16* unpacked_q, BFloat16* unpacked_k, BFloat16* unpacked_v, const int num_heads, const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size, cudaStream_t stream, const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
