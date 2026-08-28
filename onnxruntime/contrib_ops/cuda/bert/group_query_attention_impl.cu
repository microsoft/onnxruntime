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

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/utils/debug_macros.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/unfused_attention.h"
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

template <typename T>
__global__ void ConvertHeadSinkToFloatKernel(const T* input, float* output, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    output[i] = static_cast<float>(input[i]);
  }
}

template <typename T>
Status LaunchConvertHeadSinkToFloat(
    const T* input,
    float* output,
    int count,
    cudaStream_t stream,
    int max_threads_per_block) {
  int blocks = (count + max_threads_per_block - 1) / max_threads_per_block;
  ConvertHeadSinkToFloatKernel<T><<<blocks, max_threads_per_block, 0, stream>>>(input, output, count);
  return CUDA_CALL(cudaGetLastError());
}

// Standalone per-head RMS normalization (QK-Norm prologue) for Q in BSNH layout.
// Each block handles one (b, s, head) vector and normalizes over head_size:
//   out[c] = in[c] * rsqrt(mean(in^2) + epsilon) * weight[c]
// where weight has shape (head_size,) and is shared across heads. This is used by the shared-KV
// (kv_sequence_length == 0) path, which processes Q only; the new-KV path folds the equivalent
// normalization into UnpackRoPEAppend before RoPE.
template <typename T>
__global__ void PerHeadRMSNormBSNHKernel(
    T* output, const T* input, const T* weight, const int head_size, const float epsilon) {
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int n = blockIdx.z;
  const int sequence_length = gridDim.x;
  const int num_heads = gridDim.z;
  const int i = threadIdx.x;

  const int64_t base = (((static_cast<int64_t>(b) * sequence_length + s) * num_heads) + n) * head_size;

  extern __shared__ float s_sumsq[];
  const float x = (i < head_size) ? static_cast<float>(input[base + i]) : 0.0f;
  s_sumsq[i] = x * x;
  __syncthreads();

  // Tree reduction over a power-of-two block size; padded entries (i >= head_size) are zero.
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (i < stride) {
      s_sumsq[i] += s_sumsq[i + stride];
    }
    __syncthreads();
  }

  if (i < head_size) {
    const float inv_rms = rsqrtf(s_sumsq[0] / static_cast<float>(head_size) + epsilon);
    output[base + i] = static_cast<T>(x * inv_rms * static_cast<float>(weight[i]));
  }
}

template <typename T>
Status LaunchPerHeadRMSNorm(
    cudaStream_t stream, T* output, const T* input, const T* weight, const float epsilon,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const int max_threads_per_block) {
  // Round the thread count up to a power of two so the tree reduction is exact.
  int tpb = 1;
  while (tpb < head_size) {
    tpb <<= 1;
  }
  if (tpb > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "head_size (", head_size, ") exceeds max threads for QK-Norm.");
  }
  const dim3 grid(sequence_length, batch_size, num_heads);
  const dim3 block(tpb);
  const size_t smem = static_cast<size_t>(tpb) * sizeof(float);
  PerHeadRMSNormBSNHKernel<T><<<grid, block, smem, stream>>>(output, input, weight, head_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

// Internal helper to get Q, K, V pointers, handling packed input
//
// Upper bound (exclusive) for absolute RoPE positions. The cos/sin caches are the real constraint;
// fall back to the KV cache sequence dimension when rotary caches are absent so behaviour is
// unchanged for models that do not use rotary embeddings. With a windowed KV cache these two differ
// by construction: RoPE stays in absolute coordinates while the cache is capacity-relative.
inline int RopeMaxPosition(const GroupQueryAttentionParameters& parameters) {
  return parameters.rotary_max_position > 0 ? parameters.rotary_max_position
                                            : parameters.seqlen_present_kv_cache;
}

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
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  T* q_out = reinterpret_cast<T*>(data.qkv_buffer);

  if (!parameters.is_packed_qkv && !parameters.do_rotary && !parameters.use_qk_norm) {
    q_out = nullptr;
  }

  U* k = reinterpret_cast<U*>(data.present_key);
  U* v = reinterpret_cast<U*>(data.present_value);
  int max_cache_length = parameters.seqlen_present_kv_cache;

  if (!parameters.past_present_share_buffer) {
    size_t kv_buffer_size = (size_t)batch_size * kv_num_heads * max_cache_length * head_size * sizeof(U);
    CUDA_CALL_THROW(cudaMemsetAsync(data.present_key, 0, kv_buffer_size, stream));
    CUDA_CALL_THROW(cudaMemsetAsync(data.present_value, 0, kv_buffer_size, stream));
  }

  bool is_cache_bnsh = (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  assert(is_cache_bnsh);  // Only support BNSH format for now

  // Copy past KV to present KV if needed
  if (!parameters.past_present_share_buffer && data.past_key != nullptr && parameters.seqlen_past_kv_cache > 0) {
    size_t src_pitch = (size_t)parameters.seqlen_past_kv_cache * head_size * sizeof(U);
    size_t dst_pitch = (size_t)max_cache_length * head_size * sizeof(U);
    size_t width = src_pitch;
    size_t height = (size_t)batch_size * kv_num_heads;
    CUDA_CALL_THROW(cudaMemcpy2DAsync(data.present_key, dst_pitch, data.past_key, src_pitch, width, height,
                                      cudaMemcpyDeviceToDevice, stream));
    CUDA_CALL_THROW(cudaMemcpy2DAsync(data.present_value, dst_pitch, data.past_value, src_pitch, width, height,
                                      cudaMemcpyDeviceToDevice, stream));
  }

  // Shared KV path: K/V inputs are empty (kv_sequence_length == 0) and the
  // past buffer already contains the full shared KV cache.  This requires
  // past_key/past_value to be provided (with RoPE already applied to K).
  // When past_present_share_buffer is true, present aliases past and no copy
  // is needed.  When false (e.g., first prompt), the past→present memcpy
  // above has already populated the present buffer with the shared KV data.
  // In both cases, only Q processing (RoPE if configured) is needed here.
  if (kv_sequence_length == 0) {
    // QK-Norm: normalize Q (BSNH) into q_out before RoPE. K is already normalized in the shared cache
    // (it was normalized when first appended), so only Q needs processing on this path.
    const T* q_rope_input = data.query;
    if (parameters.use_qk_norm) {
      ORT_RETURN_IF_ERROR((LaunchPerHeadRMSNorm<T>(
          stream, q_out, data.query, data.q_norm_weight, parameters.qk_norm_epsilon,
          batch_size, sequence_length, num_heads, head_size, max_threads_per_block)));
      // RoPE (if any) runs in-place on the normalized Q; the rotary kernel handles in-place safely.
      q_rope_input = q_out;
    }
    if (parameters.do_rotary && data.cos_cache != nullptr && data.sin_cache != nullptr) {
      // Apply RoPE to Q only using the standalone rotary embedding kernel.
      // Q is in BSNH format; the kernel writes rotated Q to q_out.
      // position_ids_format: 1 = explicit per-token position_ids, 2 = past_seq_lens + s
      // When position_ids is null, use format 2 (derives position from past_seq_lens).
      const int pos_format = data.position_ids != nullptr ? 1 : 2;
      if constexpr (std::is_same<T, __half>::value) {
        ORT_RETURN_IF_ERROR((LaunchRotaryEmbeddingKernel<half>(
            stream, reinterpret_cast<half*>(q_out), reinterpret_cast<const half*>(q_rope_input),
            data.position_ids, data.past_seq_lens,
            reinterpret_cast<const half*>(data.cos_cache), reinterpret_cast<const half*>(data.sin_cache),
            batch_size, sequence_length, num_heads, head_size, parameters.rotary_dim, RopeMaxPosition(parameters),
            pos_format, parameters.rotary_interleaved,
            max_threads_per_block, false /* is_input_bnsh_format: Q is BSNH */)));
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        ORT_RETURN_IF_ERROR((LaunchRotaryEmbeddingKernel<onnxruntime::BFloat16>(
            stream, reinterpret_cast<onnxruntime::BFloat16*>(q_out), reinterpret_cast<const onnxruntime::BFloat16*>(q_rope_input),
            data.position_ids, data.past_seq_lens,
            reinterpret_cast<const onnxruntime::BFloat16*>(data.cos_cache), reinterpret_cast<const onnxruntime::BFloat16*>(data.sin_cache),
            batch_size, sequence_length, num_heads, head_size, parameters.rotary_dim, RopeMaxPosition(parameters),
            pos_format, parameters.rotary_interleaved,
            max_threads_per_block, false /* is_input_bnsh_format: Q is BSNH */)));
      }
    }
    // If do_rotary is false and QK-Norm is off, Q is used directly from data.query (q_out == nullptr).
    // K/V present buffers already point to the shared past — no work needed.
  } else {
    ORT_RETURN_IF_ERROR((LaunchUnpackRoPEAppend<T, U>(
        parameters.is_packed_qkv ? reinterpret_cast<const T*>(data.query) : nullptr,
        parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.query),
        parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.key),
        parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.value),
        q_out, k, v, data.k_scale, data.v_scale, nullptr /* q_fold_scale */,
        num_heads, kv_num_heads, head_size, sequence_length, batch_size,
        RopeMaxPosition(parameters), max_cache_length,
        data.past_seq_lens, data.cache_past_seq_lens,
        reinterpret_cast<const T*>(data.cos_cache), reinterpret_cast<const T*>(data.sin_cache),
        parameters.rotary_dim, data.position_ids, parameters.rotary_interleaved,
        is_cache_bnsh, parameters.k_quant_type,
        data.q_norm_weight, data.k_norm_weight, parameters.qk_norm_epsilon,
        stream, max_threads_per_block)));
  }

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

// Offset one past the last resident row of a windowed KV cache, i.e. where the next entry is
// appended. Whole `gap`-sized blocks are reclaimed at once, which keeps the result in
// [capacity - gap + 1, capacity] once the cache has filled:
//   end(P) = P while P <= capacity, else P - gap * ceil((P - capacity) / gap)
//
// This mirrors WindowedCacheEnd() in the CPU kernel (contrib_ops/cpu/bert/group_query_attention.cc);
// the two layouts must agree or CPU/CUDA parity breaks, so keep them in sync.
//
// The intermediate math runs in int64_t because `past_sequence_length` derives from the
// caller-supplied seqlens_k input: `overflow + gap - 1` and `gap * blocks` could otherwise overflow
// for values near INT32_MAX, and signed overflow is UB. The result is bounded by `capacity`, so
// narrowing back to int is safe.
__device__ __forceinline__ int WindowedCacheEnd(int64_t past_sequence_length, int64_t capacity, int64_t gap) {
  if (past_sequence_length <= capacity) {
    return static_cast<int>(past_sequence_length);  // still filling: nothing has been reclaimed yet
  }
  const int64_t overflow = past_sequence_length - capacity;
  const int64_t reclaimed = gap * ((overflow + gap - 1) / gap);
  return static_cast<int>(past_sequence_length - reclaimed);
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
                                   int* cache_past_seq_lens,
                                   int* cache_total_seq_lens,
                                   int* evict_counts,
                                   const int batch_size,
                                   const int sequence_length,
                                   const bool is_first_prompt,
                                   const int kv_cache_capacity,
                                   const int kv_cache_real_capacity) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < batch_size) {
    // total_seq_lens_minus_one is the seqlens_k input and is not range-checked on the device.
    // Clamp the negative case at the source so the derived lengths below stay non-negative and
    // cannot flow as negative offsets into KV-cache or attention index computations.
    const int seqlens_k = total_seq_lens_minus_one[i];
    const int total_len = (seqlens_k > 0 ? seqlens_k : 0) + 1;
    total_seq_lens[i] = total_len;
    int past_len;
    if (is_first_prompt) {
      past_len = 0;
      past_seq_lens[i] = 0;
      padded_seq_lens[i] = sequence_length;
    } else {
      past_len = total_len - sequence_length;
      past_len = past_len > 0 ? past_len : 0;
      past_seq_lens[i] = past_len;
      padded_seq_lens[i] = 0;
    }

    if (cache_past_seq_lens == nullptr) {
      return;
    }

    // Cache-relative coordinates for a windowed KV cache. The cache holds the L = min(T, C) most
    // recent tokens contiguously at indices [0, L), where C is the real capacity.
    //   Lp = min(P, C)                  tokens currently resident
    //   E  = max(0, Lp + S - C)         tokens this append pushes out of the real cache
    //   D  = max(0, Lp + S - C_used)    tokens that have to be shifted out of the buffer in use
    // A multi-token step writes into a staging buffer long enough to hold Lp + S entries, so there
    // D is 0 and E is only the offset of the window inside the staging buffer. For a single-token
    // step the staging buffer is the cache itself (C_used == C) and D == E.
    //
    // For single-token decode steps the drifting-layout optimisation applies: entries live at
    // [0, end) and are appended at `end`; compaction reclaims a whole block of `gap = C - W + 1`
    // rows at once, so the kernel is load-bearing on only one step in `gap` and on the remaining
    // steps the d <= 0 early-exit fires. Note that `end` is still a pure function of the absolute
    // past length P, so the op stays stateless across Run() calls.
    // For multi-token steps the staging buffer size must be predictable from Lp = min(P, C), and
    // the write-back offset is the number of evicted rows under the old formula, so we keep the
    // original Lp-based computation for that case.
    const int C = kv_cache_real_capacity;
    const int C_used = kv_cache_capacity;
    const int W = C < C_used ? C : C_used;  // == local_window_size when C == C_used
    const int gap = C - W + 1;              // >= 1; at C == W (no slack) this is 1 and every step compacts

    int E, D;
    int Lp;
    if (sequence_length == 1) {
      // Drifting-layout end pointer for single-token steps; see WindowedCacheEnd above.
      // This keeps W <= end <= C once the cache has filled.
      const int end_before = WindowedCacheEnd(past_len, C, gap);
      const int end_after = WindowedCacheEnd(static_cast<int64_t>(past_len) + 1, C, gap);
      const int kept = end_after - 1;
      E = end_before - kept > 0 ? end_before - kept : 0;
      D = (C_used < C) ? 0 : E;
      Lp = end_before;
    } else {
      // Original formula for multi-token (staging) steps: Lp = min(P, C), E = evicted rows.
      Lp = past_len < C ? past_len : C;
      const int evicted = Lp + sequence_length - C;
      const int shifted = Lp + sequence_length - C_used;
      E = evicted > 0 ? evicted : 0;
      D = shifted > 0 ? shifted : 0;
    }
    cache_past_seq_lens[i] = Lp - D;
    cache_total_seq_lens[i] = (Lp + sequence_length) < C_used ? (Lp + sequence_length) : C_used;
    evict_counts[i] = E;
  }
}

Status LaunchGetSequenceLengths(
    const int* total_seq_lens_minus_one,
    int* past_seq_lens,
    int* total_seq_lens,
    int* padded_seq_lens,
    int* cache_past_seq_lens,
    int* cache_total_seq_lens,
    int* evict_counts,
    const int batch_size,
    const int sequence_length,
    const bool is_first_prompt,
    const int kv_cache_capacity,
    const int kv_cache_real_capacity,
    cudaStream_t stream,
    const int max_threads_per_block) {
  int blocks = (batch_size + max_threads_per_block - 1) / max_threads_per_block;
  GetSequenceLengths<<<blocks, max_threads_per_block, 0, stream>>>(
      total_seq_lens_minus_one, past_seq_lens, total_seq_lens, padded_seq_lens,
      cache_past_seq_lens, cache_total_seq_lens, evict_counts,
      batch_size, sequence_length, is_first_prompt, kv_cache_capacity, kv_cache_real_capacity);
  return CUDA_CALL(cudaGetLastError());
}

// ============================================================================
// Windowed KV cache compaction
// ============================================================================
// Left-shifts a BNSH KV cache by D = evict_counts[b] positions so that the retained tokens stay
// contiguous at cache indices [0, Lp - D). D is read on the device (it changes every step once the
// cache is full), while the launch geometry is sized for the worst case and threads whose source
// row is out of range exit early. Keeping the grid shape independent of D is what allows the
// enclosing graph to remain capturable.
//
// A direct in-place left shift races (source and destination rows overlap across blocks), so the
// shift runs as copy-to-scratch followed by copy-back. Both kernels operate on whole rows of the
// raw storage type, which makes them quantization agnostic (fp16/bf16/int8/fp8, and int4 as long
// as the row size is expressed in packed bytes).
//
// row_bytes: number of bytes occupied by one (batch, head, position) entry.
// capacity : allocated sequence dimension C of the cache.
// ============================================================================

// Shared launch geometry for the two row-copy kernels below. threadIdx.x walks the uint4 chunks of
// a row and threadIdx.y walks rows, so a warp covers several adjacent rows in full and the accesses
// stay coalesced. Handing a block many rows keeps the grid to a few dozen blocks, which matters
// because these copies are launch bound rather than bandwidth bound: with gap = C - W + 1,
// evict_counts[b] is non-zero on only one step in `gap`, so nearly every launch hits the d <= 0
// early-exit and its cost is dominated by grid scheduling rather than memory traffic.
static void KvRowCopyLaunchConfig(const int vec_count,
                                  const int rows,
                                  const int kv_num_heads,
                                  const int batch_size,
                                  dim3& grid,
                                  dim3& block) {
  constexpr int kThreadsPerBlock = 256;
  const int threads_x = vec_count < 32 ? vec_count : 32;
  const int threads_y = kThreadsPerBlock / threads_x > 0 ? kThreadsPerBlock / threads_x : 1;
  block = dim3(threads_x, threads_y);
  grid = dim3((rows + threads_y - 1) / threads_y, kv_num_heads, batch_size);
}

// K and V shift in the same launch: they share every index this kernel computes, so fusing them
// halves the number of launches per layer without adding a single block.
__global__ void CompactKvCacheRows(uint4* __restrict__ dst_k,
                                   const uint4* __restrict__ src_k,
                                   uint4* __restrict__ dst_v,
                                   const uint4* __restrict__ src_v,
                                   const int* __restrict__ evict_counts,
                                   const int* __restrict__ cache_past_seq_lens,
                                   const int capacity,
                                   const int vec_count,
                                   const bool copy_v,
                                   const bool src_is_shifted) {
  // grid: (ceil(capacity / blockDim.y), kv_num_heads, batch_size); block: (row chunks, rows).
  const int b = blockIdx.z;
  const int d = evict_counts[b];
  if (d <= 0) {
    return;
  }
  const int s = blockIdx.x * blockDim.y + threadIdx.y;
  if (s >= cache_past_seq_lens[b]) {  // cache_past_seq_lens[b] is the retained length Lp - D
    return;
  }

  const int64_t head_stride = static_cast<int64_t>(capacity) * vec_count;
  const int64_t base = (static_cast<int64_t>(b) * gridDim.y + blockIdx.y) * head_stride;

  // Pass 1 (src_is_shifted): scratch[s] <- cache[s + D].
  // Pass 2:                  cache[s]   <- scratch[s].
  const int64_t dst_row = base + static_cast<int64_t>(s) * vec_count;
  const int64_t src_row = base + static_cast<int64_t>(src_is_shifted ? (s + d) : s) * vec_count;

  for (int c = threadIdx.x; c < vec_count; c += blockDim.x) {
    dst_k[dst_row + c] = src_k[src_row + c];
    if (copy_v) {
      dst_v[dst_row + c] = src_v[src_row + c];
    }
  }
}

// Evicts the oldest evict_counts[b] entries from a windowed KV cache by shifting the retained
// entries to the front. Must run before the new tokens are appended.
Status LaunchCompactKvCache(void* k_cache,
                            void* v_cache,
                            void* scratch,
                            const int* evict_counts,
                            const int* cache_past_seq_lens,
                            const int batch_size,
                            const int kv_num_heads,
                            const int capacity,
                            const int row_bytes,
                            cudaStream_t stream) {
  if (capacity <= 0 || row_bytes <= 0) {
    return Status::OK();
  }
  if (row_bytes % static_cast<int>(sizeof(uint4)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "sliding_window_cache: KV cache row size (", row_bytes,
                           " bytes) must be a multiple of 16 for the compaction shift.");
  }

  // Order the two caches so the fused kernel always has a valid first operand; the second one is
  // optional, so a cache pair with only one side present still works.
  uint8_t* cache_a = reinterpret_cast<uint8_t*>(k_cache != nullptr ? k_cache : v_cache);
  if (cache_a == nullptr) {
    return Status::OK();
  }
  uint8_t* cache_b = reinterpret_cast<uint8_t*>((k_cache != nullptr && v_cache != nullptr) ? v_cache : nullptr);

  const size_t cache_bytes = static_cast<size_t>(batch_size) * kv_num_heads * capacity * row_bytes;
  ORT_RETURN_IF(scratch == nullptr && cache_bytes > 0,
                "sliding_window_cache: compaction scratch buffer was not allocated.");
  // The two caches cannot share one scratch region: their passes now run concurrently. The buffer
  // is allocated with room for both.
  uint8_t* scratch_a = reinterpret_cast<uint8_t*>(scratch);
  uint8_t* scratch_b = scratch_a + cache_bytes;

  const int vec_count = row_bytes / static_cast<int>(sizeof(uint4));
  dim3 grid, block;
  KvRowCopyLaunchConfig(vec_count, capacity, kv_num_heads, batch_size, grid, block);

  CompactKvCacheRows<<<grid, block, 0, stream>>>(
      reinterpret_cast<uint4*>(scratch_a), reinterpret_cast<const uint4*>(cache_a),
      reinterpret_cast<uint4*>(scratch_b), reinterpret_cast<const uint4*>(cache_b),
      evict_counts, cache_past_seq_lens, capacity, vec_count,
      /*copy_v=*/cache_b != nullptr, /*src_is_shifted=*/true);
  CompactKvCacheRows<<<grid, block, 0, stream>>>(
      reinterpret_cast<uint4*>(cache_a), reinterpret_cast<const uint4*>(scratch_a),
      reinterpret_cast<uint4*>(cache_b), reinterpret_cast<const uint4*>(scratch_b),
      evict_counts, cache_past_seq_lens, capacity, vec_count,
      /*copy_v=*/cache_b != nullptr, /*src_is_shifted=*/false);
  return CUDA_CALL(cudaGetLastError());
}

// ============================================================================
// Windowed KV cache: copy a window of rows between two BNSH KV caches.
//
// A step with more than one new token runs against a staging cache long enough to hold the
// resident history plus the new tokens, so that queries can still reach every key their window
// covers. This kernel both seeds that staging cache from the real one and copies the surviving
// tail back afterwards. src_offsets is a device vector (one entry per batch) because the offset of
// the surviving tail depends on the past length, which is only known on the device; passing it
// through memory keeps the launch geometry constant and therefore CUDA-graph capturable.
// The copy works on raw rows, which keeps it independent of the cache element type and of whether
// the cache is written in quantized form. K and V share every index, so they move together.
// ============================================================================
__global__ void CopyKvCacheRows(uint4* __restrict__ dst_k,
                                const uint4* __restrict__ src_k,
                                uint4* __restrict__ dst_v,
                                const uint4* __restrict__ src_v,
                                const int* __restrict__ src_offsets,
                                const int src_capacity,
                                const int dst_capacity,
                                const int rows,
                                const int vec_count,
                                const bool copy_v) {
  // grid: (ceil(rows / blockDim.y), kv_num_heads, batch_size); block: (row chunks, rows).
  const int b = blockIdx.z;
  const int s = blockIdx.x * blockDim.y + threadIdx.y;
  if (s >= rows || s >= dst_capacity) {
    return;
  }
  const int src_index = s + (src_offsets == nullptr ? 0 : src_offsets[b]);
  if (src_index < 0 || src_index >= src_capacity) {
    return;
  }

  const int64_t head = static_cast<int64_t>(b) * gridDim.y + blockIdx.y;
  const int64_t src_row = (head * src_capacity + src_index) * vec_count;
  const int64_t dst_row = (head * dst_capacity + s) * vec_count;

  for (int c = threadIdx.x; c < vec_count; c += blockDim.x) {
    dst_k[dst_row + c] = src_k[src_row + c];
    if (copy_v) {
      dst_v[dst_row + c] = src_v[src_row + c];
    }
  }
}

Status LaunchCopyKvCacheWindow(void* dst_k,
                               void* dst_v,
                               const void* src_k,
                               const void* src_v,
                               const int batch_size,
                               const int kv_num_heads,
                               const int src_capacity,
                               const int dst_capacity,
                               const int rows,
                               const int* src_offsets,
                               const int row_bytes,
                               cudaStream_t stream) {
  if (rows <= 0 || row_bytes <= 0) {
    return Status::OK();
  }
  if (row_bytes % static_cast<int>(sizeof(uint4)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "sliding_window_cache: KV cache row size (", row_bytes,
                           " bytes) must be a multiple of 16 to stage the KV cache.");
  }

  // Order the pairs so the first operand is always valid; the second one is optional.
  const bool has_k = src_k != nullptr && dst_k != nullptr;
  const bool has_v = src_v != nullptr && dst_v != nullptr;
  if (!has_k && !has_v) {
    return Status::OK();
  }
  const void* src_a = has_k ? src_k : src_v;
  void* dst_a = has_k ? dst_k : dst_v;
  const void* src_b = (has_k && has_v) ? src_v : nullptr;
  void* dst_b = (has_k && has_v) ? dst_v : nullptr;

  const int vec_count = row_bytes / static_cast<int>(sizeof(uint4));
  dim3 grid, block;
  KvRowCopyLaunchConfig(vec_count, rows, kv_num_heads, batch_size, grid, block);

  CopyKvCacheRows<<<grid, block, 0, stream>>>(
      reinterpret_cast<uint4*>(dst_a), reinterpret_cast<const uint4*>(src_a),
      reinterpret_cast<uint4*>(dst_b), reinterpret_cast<const uint4*>(src_b),
      src_offsets, src_capacity, dst_capacity, rows, vec_count, /*copy_v=*/src_b != nullptr);
  return CUDA_CALL(cudaGetLastError());
}

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
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);

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

  bool past_bsnh = (past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);

  // XQA only understands a scalar dequantization scale. Per-channel scales are folded out of the
  // kernel instead: sk into Q (the channel dim is contracted by Q*K.T) and sv into the attention
  // output (the channel dim is free in P*V). Both are exact -- see the derivation next to
  // LaunchScaleHeadsByChannelScale in group_query_attention_qdq.cuh.
  // The Q side is folded into the preprocess kernel below, which already loads and stores Q, so it
  // costs no extra pass; the V side needs one elementwise pass over the output after XQA returns.
  const bool fold_k_scale_into_q = parameters.k_quant_type == KVQuantizationType::PER_CHANNEL;
  const bool fold_v_scale_into_output = parameters.v_quant_type == KVQuantizationType::PER_CHANNEL;

  // Ultimate Fused Preprocessing: Unpack, RoPE Q, RoPE K, Quantize K/V, Append K/V
  // This replaces all manual steps (Rotate Q, Rotate K, Quantize, StridedCopy)
  T* q_rot_ptr = reinterpret_cast<T*>(data.qkv_buffer);
  const T* q_input_for_xqa = q_rot_ptr;
  if (q_rot_ptr == nullptr) {
    q_input_for_xqa = reinterpret_cast<const T*>(data.query);
  }
  // GQABufferRequirements reserves qkv_buffer whenever the K scale has to be folded into Q, so the
  // preprocess kernel always has somewhere to write the scaled Q (rotated or not).
  ORT_ENFORCE(!fold_k_scale_into_q || q_rot_ptr != nullptr,
              "Per-channel XQA requires a scratch buffer for the scaled Q.");

  ORT_RETURN_IF_ERROR((LaunchUnpackRoPEAppend<T, U>(
      parameters.is_packed_qkv ? reinterpret_cast<const T*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.value),
      q_rot_ptr,  // unpacked_q (can be null if !do_rotary)
      reinterpret_cast<U*>(data.present_key),
      reinterpret_cast<U*>(data.present_value),
      data.k_scale,
      data.v_scale,
      fold_k_scale_into_q ? data.k_scale : nullptr,  // q_fold_scale
      num_heads,
      kv_num_heads,
      head_size,
      sequence_length,
      batch_size,
      RopeMaxPosition(parameters),
      parameters.seqlen_present_kv_cache,  // cache capacity C
      data.past_seq_lens,
      data.cache_past_seq_lens,
      reinterpret_cast<const T*>(data.cos_cache),
      reinterpret_cast<const T*>(data.sin_cache),
      parameters.do_rotary ? parameters.rotary_dim : 0,
      data.position_ids,
      parameters.rotary_interleaved,
      !past_bsnh,  // is_cache_bnsh
      parameters.k_quant_type,
      data.q_norm_weight, data.k_norm_weight, parameters.qk_norm_epsilon,
      stream,
      device_prop.maxThreadsPerBlock)));

  // Determine workspace size for XQA
  void* xqa_workspace = data.xqa_buffer;
  size_t xqa_workspace_size = data.xqa_buffer_bytes;

  if (data.xqa_head_sink_needs_conversion) {
    ORT_ENFORCE(data.xqa_head_sink != nullptr, "XQA head_sink conversion buffer was not allocated.");
    ORT_ENFORCE(data.head_sink != nullptr, "XQA head_sink input was not available for conversion.");
    ORT_RETURN_IF_ERROR(LaunchConvertHeadSinkToFloat<T>(
        data.head_sink, data.xqa_head_sink, num_heads, stream, device_prop.maxThreadsPerBlock));
  }

  constexpr bool is_fp8 = std::is_same<U, __nv_fp8_e4m3>::value;
  using onnxruntime::contrib::cuda::XqaQuantType;
  // 5. Launch XQA
  Status status = onnxruntime::contrib::cuda::LaunchXQAKernel<T>(
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
      parameters.local_window_size,  // -1 means global attention; >0 enables sliding window
      past_bsnh,
      // With a windowed cache the XQA kernel must index the cache in cache-relative coordinates.
      parameters.is_windowed_kv_cache ? data.cache_past_seq_lens : data.past_seq_lens,
      data.xqa_head_sink,
      fold_k_scale_into_q ? nullptr : data.k_scale,       // k_cache_scale (null: already folded into Q)
      fold_v_scale_into_output ? nullptr : data.v_scale,  // v_cache_scale (null: applied to the output below)
      // Map cache type to XqaQuantType: NONE->kNone, Float8E4M3FN->kFp8, int8->kInt8
      (parameters.k_quant_type == KVQuantizationType::NONE) ? XqaQuantType::kNone : (is_fp8 ? XqaQuantType::kFp8 : XqaQuantType::kInt8),
      xqa_workspace,
      xqa_workspace_size);

  // If XQA launch fails, debugging info

  ORT_RETURN_IF_ERROR(status);

  if (fold_v_scale_into_output) {
    ORT_RETURN_IF_ERROR((LaunchScaleHeadsByChannelScale<T>(
        stream, data.output, data.output, data.v_scale,
        batch_size, sequence_length, num_heads, kv_num_heads, head_size)));
  }

  return Status::OK();
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
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);
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
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value;

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
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);

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
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value;

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
  // A windowed cache is indexed in cache-relative coordinates, so the attention length is the
  // number of resident entries rather than the absolute total sequence length.
  int* seq_lens = parameters.is_windowed_kv_cache
                      ? data.cache_total_seq_lens
                      : (parameters.is_first_prompt ? data.padded_seq_lens : data.total_seq_lens);

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
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);

  assert(!parameters.is_first_prompt);  // Only support first prompt for this function.
  assert(parameters.k_quant_type != KVQuantizationType::NONE || parameters.v_quant_type != KVQuantizationType::NONE);

  ORT_GQA_TRACE("DequantizeFlashAttentionFallback");

  // We need to dequantize the entire KV cache (present_key/value) into a float/half buffer (data.qkv_buffer).
  // Layout in qkv_buffer: [Q (rotated)] [K_dequantized] [V_dequantized]

  T* q_rot = reinterpret_cast<T*>(data.qkv_buffer);
  size_t q_elements = static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads * parameters.head_size;
  size_t k_elements = static_cast<size_t>(parameters.batch_size) * parameters.seqlen_present_kv_cache * parameters.kv_num_heads * parameters.head_size;
  T* k_dequant = q_rot + q_elements;
  T* v_dequant = k_dequant + k_elements;

  // Step 1: Update Quantized Cache
  // We can use LaunchUnpackRoPEQuantizeAppend to unpack new QKV, apply RoPE, and append to quantized cache.
  // This will also put rotated Q into q_rot.
  ORT_RETURN_IF_ERROR((LaunchUnpackRoPEAppend<T, U>(
      parameters.is_packed_qkv ? reinterpret_cast<const T*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.value),
      q_rot, reinterpret_cast<U*>(data.present_key), reinterpret_cast<U*>(data.present_value),
      data.k_scale, data.v_scale, nullptr /* q_fold_scale: this path dequantizes the cache itself */,
      parameters.num_heads, parameters.kv_num_heads, parameters.head_size, parameters.sequence_length, parameters.batch_size,
      RopeMaxPosition(parameters), parameters.seqlen_present_kv_cache,
      data.past_seq_lens, data.cache_past_seq_lens,
      reinterpret_cast<const T*>(data.cos_cache), reinterpret_cast<const T*>(data.sin_cache),
      parameters.rotary_dim, data.position_ids, parameters.rotary_interleaved,
      (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH),
      parameters.k_quant_type,
      data.q_norm_weight, data.k_norm_weight, parameters.qk_norm_epsilon,
      stream, device_prop.maxThreadsPerBlock)));

  // Step 2: Dequantize Entire Cache
  // We now have the updated quantized cache in data.present_key/value. We need to dequantize it to k_dequant/v_dequant.
  // Only the rows that belong to the sequence are dequantized: flash attention below is given the
  // same total_seq_lens and never reads the padding tail of the cache, so dequantizing the full
  // (max_length sized) capacity on every decode step is pure memory traffic.
  bool is_bsnh = (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);

  ORT_RETURN_IF_ERROR((LaunchDequantizeKVPair<T, U, float>(
      stream, k_dequant, v_dequant,
      reinterpret_cast<const U*>(data.present_key), reinterpret_cast<const U*>(data.present_value),
      data.k_scale, data.v_scale, parameters.batch_size, parameters.kv_num_heads,
      parameters.seqlen_present_kv_cache, parameters.head_size, parameters.kv_cache_bit_width,
      parameters.k_quant_type, parameters.v_quant_type, is_bsnh, data.total_seq_lens)));

  // Step 3: Run Flash Attention on dequantized k/v
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value;

  // Use the total_seq_lens here since k_dequant/v_dequant has both past and new tokens.
  // With a windowed cache the dequantized buffer is capacity-relative, so use the cache-relative
  // lengths; the causal / local-window masks only depend on relative distances, so this is exact.
  void* seqlens_k_ptr = const_cast<void*>(reinterpret_cast<const void*>(
      parameters.is_windowed_kv_cache ? data.cache_total_seq_lens : data.total_seq_lens));
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

// Use Flash Attention for float key and value, then quantize key/value (int8/fp8/int4) to save to k/v cache.
template <typename T, typename U>
Status FlashAttentionAndQuantizeKV(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);
  assert(parameters.is_first_prompt);  // Only support first prompt for this function.
  assert(parameters.k_quant_type != KVQuantizationType::NONE || parameters.v_quant_type != KVQuantizationType::NONE);

  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int num_heads = parameters.num_heads;
  const int head_size = parameters.head_size;

  ORT_GQA_TRACE("FlashAttentionAndQuantizeKV");

  ORT_ENFORCE(parameters.past_kv_format != AttentionQkvFormat::Q_K_V_BSNH, "GQA only supports BNSH format for KV cache.");

  size_t q_elements = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size;
  size_t k_elements = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size;

  T* q_final = reinterpret_cast<T*>(data.qkv_buffer);

  // For FlashAttentionAndQuantizeKV, we need float K and V for attention.
  // We'll write them to qkv_buffer.
  T* k_final = q_final + q_elements;
  T* v_final = k_final + k_elements;

  ORT_RETURN_IF_ERROR((LaunchUnpackRoPEAppend<T, T>(
      parameters.is_packed_qkv ? reinterpret_cast<const T*>(data.query) : nullptr,
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.query),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.key),
      parameters.is_packed_qkv ? nullptr : reinterpret_cast<const T*>(data.value),
      q_final, k_final, v_final, nullptr, nullptr, nullptr /* q_fold_scale */,
      num_heads, kv_num_heads, head_size, sequence_length, batch_size,
      RopeMaxPosition(parameters), sequence_length,
      data.past_seq_lens, data.past_seq_lens,
      reinterpret_cast<const T*>(data.cos_cache), reinterpret_cast<const T*>(data.sin_cache),
      parameters.rotary_dim, data.position_ids, parameters.rotary_interleaved,
      false,  // BSNH for scratch
      KVQuantizationType::NONE,
      data.q_norm_weight, data.k_norm_weight, parameters.qk_norm_epsilon,
      stream, max_threads_per_block)));

  // 2. Run Float Flash Attention
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, __nv_bfloat16>::value;

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
      local_window_size,
      /*cache_batch_idx*/ nullptr, /*leftpad_k*/ nullptr,
      // head_sink must be forwarded here as well. This is the only prompt path taken when the KV
      // cache is quantized, so dropping it silently disables attention sinks for the whole prompt
      // while the unquantized prompt path (FlashAttention) keeps them.
      reinterpret_cast<void*>(const_cast<T*>(data.head_sink))));

  if (parameters.k_quant_type != KVQuantizationType::NONE) {
    ORT_RETURN_IF_ERROR((LaunchQuantizeKV<T, U, float>(
        stream, reinterpret_cast<U*>(data.present_key), reinterpret_cast<const T*>(k_final), data.k_scale,
        nullptr, data.total_seq_lens, batch_size, kv_num_heads, sequence_length, parameters.seqlen_present_kv_cache,
        head_size, parameters.kv_cache_bit_width, parameters.k_quant_type, true)));
  }

  if (parameters.v_quant_type != KVQuantizationType::NONE) {
    ORT_RETURN_IF_ERROR((LaunchQuantizeKV<T, U, float>(
        stream, reinterpret_cast<U*>(data.present_value), reinterpret_cast<const T*>(v_final), data.v_scale,
        nullptr, data.total_seq_lens, batch_size, kv_num_heads, sequence_length, parameters.seqlen_present_kv_cache,
        head_size, parameters.kv_cache_bit_width, parameters.v_quant_type, true)));
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
  static_assert(std::is_same<T, typename OrtToCudaType<T>::type>::value);
  static_assert(std::is_same<U, typename OrtToCudaType<U>::type>::value);

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
  p.is_bf16 = std::is_same<T, __nv_bfloat16>::value;
  p.is_half = !p.is_bf16 && (sizeof(T) == 2);
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.sequence_length = sequence_length;
  p.kv_sequence_length = present_sequence_length;  // maybe remove
  p.max_sequence_length = present_sequence_length;
  p.qk_head_size = head_size;
  p.v_head_size = head_size;
  p.causal = parameters.is_unidirectional;
  p.scale = scale;
  p.softcap = parameters.softcap;
  p.seqlen_k_ptr = parameters.is_windowed_kv_cache
                       ? data.cache_total_seq_lens
                       : (parameters.is_first_prompt ? data.padded_seq_lens : data.total_seq_lens);
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

// ============================================================================
// UnfusedGqaAttention: fallback path that handles GQA natively and fixes the
// fp16 head_size > 256 NaN (issue #28195).
//
// Dispatched when Flash / MEA / XQA are all ineligible. Supports:
//   - Any head_size up to H (FP32 QK accumulation avoids fp16 overflow).
//   - GQA (num_heads != kv_num_heads) via reshape-Q trick in the GEMM.
//   - Different Q / K sequence lengths (first prompt or decode with past).
//   - Causal, sliding window (local_window_size), softcap, per-batch seqlens.
//
// Not supported (caller falls through elsewhere):
//   - Quantized KV cache (U != T): hit by the original NOT_IMPLEMENTED path.
//   - Smooth softmax / head_sink: Flash-only feature.
//
// attention_bias (with dim-0/dim-1 broadcast) is supported here; this is the path
// bias-carrying GQA nodes are dispatched to, since Flash/XQA/cuDNN don't take a bias.
// ============================================================================
template <typename T, typename U>
Status UnfusedGqaAttention(
    const cudaDeviceProp& device_prop,
    cublasHandle_t cublas,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  static_assert(std::is_same<T, U>::value,
                "UnfusedGqaAttention requires non-quantized KV cache (T == U).");

  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const int max_kv = parameters.seqlen_present_kv_cache;

  ORT_RETURN_IF(data.unfused_q_bnsh == nullptr || data.unfused_y_bnsh == nullptr ||
                    data.unfused_workspace == nullptr,
                "Unfused GQA scratch buffers are not allocated.");
  ORT_RETURN_IF(parameters.past_kv_format != AttentionQkvFormat::Q_K_V_BNSH,
                "Unfused GQA fallback requires BNSH KV cache layout.");

  ORT_GQA_TRACE("UnfusedGqaAttention");

  // Step 1: unpack Q (optionally RoPE), append new K/V into present_key/value (BNSH).
  const T* q_prep = nullptr;
  ORT_RETURN_IF_ERROR((PrepareQKV<T, U>(stream, max_threads_per_block, parameters, data, q_prep)));

  // Step 2: transpose Q from BSNH (PrepareQKV output) to BNSH.
  // Transpose_BSNH_to_BNSH has overloads for half/BFloat16/float; bridge via reinterpret_cast.
  // GQA only registers half and bf16 types; guard against accidental float instantiation.
  static_assert(std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value,
                "UnfusedGqaAttention transpose only supports __half and __nv_bfloat16.");
  if constexpr (std::is_same<T, __half>::value) {
    ORT_RETURN_IF_ERROR((Transpose_BSNH_to_BNSH(batch_size, sequence_length, num_heads, head_size,
                                                reinterpret_cast<const half*>(q_prep),
                                                reinterpret_cast<half*>(data.unfused_q_bnsh),
                                                stream, max_threads_per_block)));
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    ORT_RETURN_IF_ERROR((Transpose_BSNH_to_BNSH(batch_size, sequence_length, num_heads, head_size,
                                                reinterpret_cast<const onnxruntime::BFloat16*>(q_prep),
                                                reinterpret_cast<onnxruntime::BFloat16*>(data.unfused_q_bnsh),
                                                stream, max_threads_per_block)));
  }

  // Step 3: run unfused attention with FP32 QK accumulation.
  UnfusedAttentionParams p;
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.kv_num_heads = kv_num_heads;
  p.head_size = head_size;
  ORT_ENFORCE(head_size == parameters.v_head_size || parameters.v_head_size == 0,
              "UnfusedGqaAttention requires head_size == v_head_size");
  p.v_head_size = head_size;  // GQA op has head_size == v_head_size
  p.q_sequence_length = sequence_length;
  // For the decode/prompt, data.total_seq_lens[b] <= seqlen_present_kv_cache.
  // Use seqlen_present_kv_cache as the upper bound for the GEMM and pass per-batch
  // seqlens to the softmax so positions beyond the valid length are masked.
  p.total_kv_length = parameters.total_sequence_length;
  p.max_kv_length = max_kv;
  p.broadcast_attn_bias_dim_0 = parameters.broadcast_attn_bias_dim_0;
  p.broadcast_attn_bias_dim_1 = parameters.broadcast_attn_bias_dim_1;
  p.is_causal = parameters.is_unidirectional;
  p.local_window_size = parameters.local_window_size;  // -1 disables
  p.past_kv_length = parameters.total_sequence_length - parameters.sequence_length;
  p.scale = scale;
  p.softcap = parameters.softcap;
  p.seqlens_k = data.total_seq_lens;

  if (parameters.is_windowed_kv_cache) {
    // The cache holds only the most recent min(T, C) tokens, contiguously from index 0. Causal and
    // local-window masks depend solely on query/key distance, so shifting to cache-relative
    // coordinates leaves them unchanged.
    p.total_kv_length = std::min(parameters.total_sequence_length, max_kv);
    p.past_kv_length = p.total_kv_length - parameters.sequence_length;
    p.seqlens_k = data.cache_total_seq_lens;
  }

  ORT_RETURN_IF_ERROR((LaunchUnfusedAttention<T>(
      device_prop, cublas, stream, p,
      data.unfused_q_bnsh,
      reinterpret_cast<const T*>(data.present_key),
      reinterpret_cast<const T*>(data.present_value),
      data.attention_bias,
      data.unfused_y_bnsh,
      data.unfused_workspace,
      /*output_qk=*/nullptr)));

  // Step 4: transpose output BNSH → BSNH into data.output.
  // Use p.v_head_size (== head_size per ORT_ENFORCE) for semantic correctness.
  if constexpr (std::is_same<T, __half>::value) {
    ORT_RETURN_IF_ERROR((Transpose_BNSH_to_BSNH(batch_size, sequence_length, num_heads, p.v_head_size,
                                                reinterpret_cast<const half*>(data.unfused_y_bnsh),
                                                reinterpret_cast<half*>(data.output),
                                                stream, max_threads_per_block)));
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    ORT_RETURN_IF_ERROR((Transpose_BNSH_to_BSNH(batch_size, sequence_length, num_heads, p.v_head_size,
                                                reinterpret_cast<const onnxruntime::BFloat16*>(data.unfused_y_bnsh),
                                                reinterpret_cast<onnxruntime::BFloat16*>(data.output),
                                                stream, max_threads_per_block)));
  }
  return Status::OK();
}

////////// API Functions

// ============================================================================
// CudnnSdpaAttention: cuDNN scaled-dot-product-attention (cudnn_frontend) path.
//
// Preferred on SM>=90 (Hopper/Blackwell) for non-quantized FP16/BF16 GQA. cuDNN handles
// grouped-query attention natively (no KV head expansion) and applies causal masking through the
// diagonal-band API. The present KV cache (BNSH, padded to seqlen_present_kv_cache) is passed with
// the capacity as the KV sequence length so strides match the physical buffer; a per-batch padding
// mask (total_seq_lens) ignores the unused tail. RoPE / packed-QKV unpack and the new-token append
// are performed by PrepareQKV.
//
// Not handled here (excluded by op-level eligibility, caller falls through elsewhere):
//   - Quantized KV cache (U != T), softcap, smooth softmax / head sink, sliding window.
// ============================================================================
template <typename T, typename U>
Status CudnnSdpaAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    Stream* ort_stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data,
    float scale) {
  if constexpr (!std::is_same<T, U>::value) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "cuDNN SDPA GQA path requires a non-quantized KV cache (T == U).");
  } else {
    ORT_GQA_TRACE("CudnnSdpaAttention");

    const int max_threads_per_block = device_prop.maxThreadsPerBlock;

    // Prepare Q (optional RoPE) and append the new K/V into the present cache (BNSH, padded to
    // seqlen_present_kv_cache). After this, present_key/value hold the full KV history.
    const T* q_prep = nullptr;
    ORT_RETURN_IF_ERROR((PrepareQKV<T, U>(stream, max_threads_per_block, parameters, data, q_prep)));

    ORT_RETURN_IF(data.cudnn_handle == nullptr || data.allocator == nullptr,
                  "cuDNN SDPA GQA path is missing the cuDNN handle or temp-space allocator.");

    constexpr bool is_bf16 = std::is_same<T, __nv_bfloat16>::value;

    // First prompt may right-pad the query, so its valid query length equals total_seq_lens.
    // Otherwise every one of the sequence_length query tokens is valid and the wrapper synthesizes a
    // full-length (no-op) query padding mask.
    int* seq_len_q = parameters.is_first_prompt ? data.total_seq_lens : nullptr;
    int* seq_len_kv = data.total_seq_lens;

    ::onnxruntime::cudnn_sdpa::run(
        reinterpret_cast<void*>(data.output),
        const_cast<void*>(reinterpret_cast<const void*>(q_prep)),
        reinterpret_cast<void*>(data.present_key),
        reinterpret_cast<void*>(data.present_value),
        /*attn_bias=*/nullptr,
        seq_len_q,
        seq_len_kv,
        parameters.batch_size,
        parameters.num_heads,                // num_heads_q
        parameters.kv_num_heads,             // num_heads_kv
        parameters.head_size,                // head_size_qk
        parameters.head_size,                // head_size_v (GQA: v_head_size == head_size)
        parameters.sequence_length,          // sequence_length_q
        parameters.seqlen_present_kv_cache,  // sequence_length_kv (capacity, matches buffer strides)
        scale,
        parameters.is_unidirectional,
        is_bf16,
        /*broadcast_attn_bias_dim_0=*/false,
        /*broadcast_attn_bias_dim_1=*/false,
        /*sliding_window=*/0,
        AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH,
        static_cast<cudnnHandle_t>(data.cudnn_handle),
        ort_stream,
        data.allocator);

    return Status::OK();
  }
}

template <typename T, typename U>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;

  if (parameters.is_windowed_kv_cache && parameters.kv_cache_capacity == parameters.kv_cache_real_capacity) {
    // Evict the oldest rows first so the retained window is contiguous at [0, L) and the new
    // tokens can simply be appended after it. Skipped when ComputeInternal has redirected this
    // step through a staging cache: there the buffer is long enough that nothing is pushed out.
    const int row_bytes = (parameters.kv_cache_bit_width == 0)
                              ? static_cast<int>(parameters.head_size * sizeof(U))
                              : parameters.head_size * parameters.kv_cache_bit_width / 8;
    ORT_RETURN_IF_ERROR(LaunchCompactKvCache(
        data.present_key, data.present_value, data.compaction_scratch,
        data.evict_counts, data.cache_past_seq_lens,
        parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache,
        row_bytes, stream));
  }

  if (data.use_xqa) {
    return ExtremeDecoding(device_prop, stream, parameters, data, scale);
  }

  if (data.use_cudnn_sdpa) {
    return CudnnSdpaAttention<T, U>(device_prop, stream, ort_stream, parameters, data, scale);
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

  if (data.use_unfused) {
    if constexpr (std::is_same<T, U>::value) {
      return UnfusedGqaAttention<T, U>(device_prop, cublas, stream, parameters, data, scale);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Unfused GQA fallback does not support quantized KV cache.");
    }
  }

  if (parameters.k_quant_type != KVQuantizationType::NONE ||
      parameters.v_quant_type != KVQuantizationType::NONE) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "No available GroupQueryAttention kernel supports the quantized KV cache.");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unfused Group Query Attention not implemented yet.");
}

template struct GroupQueryAttentionData<half, half>;
template struct GroupQueryAttentionData<__nv_bfloat16, __nv_bfloat16>;
template struct GroupQueryAttentionData<half, int8_t>;

template Status LaunchConvertHeadSinkToFloat<half>(
    const half* input,
    float* output,
    int count,
    cudaStream_t stream,
    int max_threads_per_block);

template Status LaunchConvertHeadSinkToFloat<__nv_bfloat16>(
    const __nv_bfloat16* input,
    float* output,
    int count,
    cudaStream_t stream,
    int max_threads_per_block);

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

#ifdef USE_INT4_KV_CACHE
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
#endif

#ifdef USE_FP8_KV_CACHE
template struct GroupQueryAttentionData<half, __nv_fp8_e4m3>;

template Status QkvToContext<half, __nv_fp8_e4m3>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<half, __nv_fp8_e4m3>& data);

template struct GroupQueryAttentionData<__nv_bfloat16, __nv_fp8_e4m3>;

template Status QkvToContext<__nv_bfloat16, __nv_fp8_e4m3>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<__nv_bfloat16, __nv_fp8_e4m3>& data);
#endif

// Explicit instantiations for cross-TU usage by core/providers/cuda/llm/attention.cc
template Status LaunchUngroup<__half>(
    const GroupQueryAttentionParameters& parameters,
    float2* k_buff, float2* v_buff,
    const float2* k_og, const float2* v_og,
    const int buff_seqlen, const int og_seqlen,
    const bool is_bsnh,
    cudaStream_t stream,
    const int max_threads_per_block);
template Status LaunchUngroup<__nv_bfloat16>(
    const GroupQueryAttentionParameters& parameters,
    float2* k_buff, float2* v_buff,
    const float2* k_og, const float2* v_og,
    const int buff_seqlen, const int og_seqlen,
    const bool is_bsnh,
    cudaStream_t stream,
    const int max_threads_per_block);

template Status LaunchUnpackQKV<half, LAYOUT_BNSH>(const half* packed_qkv, half* unpacked_q, half* unpacked_k, half* unpacked_v, const int num_heads, const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size, cudaStream_t stream, const int max_threads_per_block);
template Status LaunchUnpackQKV<__nv_bfloat16, LAYOUT_BNSH>(const __nv_bfloat16* packed_qkv, __nv_bfloat16* unpacked_q, __nv_bfloat16* unpacked_k, __nv_bfloat16* unpacked_v, const int num_heads, const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size, cudaStream_t stream, const int max_threads_per_block);

// BFloat16 variant is used in sparse attention.
template Status LaunchUnpackQKV<BFloat16, LAYOUT_BNSH>(const BFloat16* packed_qkv, BFloat16* unpacked_q, BFloat16* unpacked_k, BFloat16* unpacked_v, const int num_heads, const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size, cudaStream_t stream, const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
