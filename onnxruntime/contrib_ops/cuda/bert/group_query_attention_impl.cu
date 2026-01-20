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
#include <cstdlib>  // For getenv

#include <cassert>
#include <cub/cub.cuh>

#include "contrib_ops/cpu/utils/debug_macros.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "contrib_ops/cuda/bert/rotary_common.cuh"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"

using namespace onnxruntime::cuda;

using onnxruntime::contrib::GroupQueryAttentionParameters;
using onnxruntime::contrib::LAYOUT_BNSH;
using onnxruntime::contrib::cuda::GroupQueryAttentionData;

namespace onnxruntime {
namespace contrib {
namespace cuda {

////////// Auxiliary Kernels for KV prep

// Concat new to past in present. Supports past BSNH or past BNSH
template <typename T>
Status LaunchConcatNewToPastKVHelper(GroupQueryAttentionParameters& parameters,
                                     GroupQueryAttentionData<T>& data,
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
                                 data.past_key,
                                 data.past_value,
                                 reinterpret_cast<const T*>(new_key),
                                 reinterpret_cast<const T*>(new_value),
                                 data.present_key,
                                 data.present_value,
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
template <typename T>
Status LaunchConcatKVInPlace(GroupQueryAttentionParameters& parameters,
                             GroupQueryAttentionData<T>& data,
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
                               data.present_key,
                               data.present_value,
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

// Fused kernel: Unpack QKV + Apply RoPE to Q and K + Append K/V directly to cache
// This eliminates 4 kernel launches: Unpack -> Rotate Q -> Rotate K -> Append K -> Append V
// Becomes: Single kernel that does all operations in one pass
//
// Bounds Safety:
//   - cache_s = past_seq_len + s is guaranteed < max_seqlen by the caller (group_query_attention.cc)
//     because present_sequence_length = max(past + new_seq_len) across batches, and the present
//     buffer is allocated with seqlen_present_kv_cache >= total_seq_lens[b] for all b.
//   - The kernel processes exactly batch_size * sequence_length * (Q+K+V hidden) elements,
//     which matches the packed_qkv input size allocated by the model.
//
// RoPE Contiguity Requirement:
//   - packed_qkv MUST be strictly contiguous with layout [B, S, (H_q + 2*H_kv) * D]
//   - The half-split RoPE logic (RotaryDispatcher::apply) fetches pair elements at offset
//     (h + rotary_dim/2) relative to the start of each head
//   - If strided/non-contiguous inputs are ever supported, this pointer arithmetic must change
//
// Performance Optimization:
//   Uses 3D grid layout to eliminate expensive integer divisions:
//   - blockIdx.z = batch index (b)
//   - blockIdx.y = sequence index (s)
//   - blockIdx.x * blockDim.x + threadIdx.x = offset within QKV hidden dimension
//   This removes 4 divisions (/, %) per thread that would otherwise be needed.
template <typename T>
__global__ void UnpackQKVWithRoPEAndAppendKV(
    const T* packed_qkv,  // Input: packed QKV [B, S, (Q+K+V) hidden]
    T* unpacked_q,        // Output: rotated Q [B, S, Q_heads, H] (BSNH)
    T* k_cache,           // Output: K cache [B, N, MaxS, H] or [B, MaxS, N, H]
    T* v_cache,           // Output: V cache [B, N, MaxS, H] or [B, MaxS, N, H]
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int d,           // QKV hidden stride = (num_heads + 2*kv_num_heads) * head_size
    const int max_seqlen,  // KV cache max sequence length
    const int* past_seq_lens,
    // RoPE params
    const T* cos_cache,
    const T* sin_cache,
    const int rotary_dim,
    const int64_t* position_ids,
    const bool interleaved,
    const bool is_cache_bnsh) {
  // Vectorized load/store using float4 (16 bytes)
  using LoadT = float4;
  constexpr int elements_per_thread = sizeof(LoadT) / sizeof(T);

  // 3D grid layout eliminates integer division:
  // - blockIdx.z = batch index (b) - obtained from grid dimension, no division needed
  // - blockIdx.y = sequence index (s) - obtained from grid dimension, no division needed
  // - linear thread index within (b, s) gives offset directly
  const int b = blockIdx.z;
  const int s = blockIdx.y;
  const int offset_vec_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Vector index within d
  const int offset = offset_vec_idx * elements_per_thread;           // Element offset within d

  // Bounds check: offset must be within the QKV hidden dimension
  if (offset >= d) return;

  const int q_hidden = num_heads * head_size;
  const int k_hidden = kv_num_heads * head_size;
  const int sequence_length = gridDim.y;  // Get from grid dimension

  // Calculate linear index for packed_qkv load
  const int64_t packed_idx = static_cast<int64_t>(b) * sequence_length * d +
                             static_cast<int64_t>(s) * d + offset;

  // Load vector from packed buffer
  LoadT val_vec = reinterpret_cast<const LoadT*>(packed_qkv)[packed_idx / elements_per_thread];

  // Common RoPE Calculations
  const int past_seq_len = past_seq_lens[b];
  int pos_id = 0;
  if (position_ids != nullptr) {
    pos_id = static_cast<int>(position_ids[b * sequence_length + s]);
  } else {
    pos_id = past_seq_len + s;
  }

  // Determine Q, K, or V based on offset
  if (offset < q_hidden) {
    // Q: Apply RoPE and write to unpacked_q buffer (BSNH format)
    const int q_head_idx = offset / head_size;
    const int h = offset % head_size;
    const int h_idx = h / elements_per_thread;

    if (cos_cache != nullptr && rotary_dim > 0 && h < rotary_dim) {
      // For half-split RoPE, pair values should be read relative to the START of the current Q head.
      // Calculate offset to head start: (b, s, q_head_n, 0) in packed QKV.
      const int64_t q_head_start_in_packed = static_cast<int64_t>(b) * sequence_length * d +
                                             static_cast<int64_t>(s) * d +
                                             static_cast<int64_t>(q_head_idx) * head_size;
      RotaryDispatcher<LoadT, T>::apply(val_vec,
                                        reinterpret_cast<const LoadT*>(cos_cache),
                                        reinterpret_cast<const LoadT*>(sin_cache),
                                        rotary_dim, h_idx, pos_id, interleaved,
                                        reinterpret_cast<const LoadT*>(packed_qkv),
                                        q_head_start_in_packed / elements_per_thread);
    }

    const int64_t q_idx = static_cast<int64_t>(b) * sequence_length * num_heads * head_size +
                          static_cast<int64_t>(s) * num_heads * head_size + offset;
    // Vector store to unpacked_q
    reinterpret_cast<LoadT*>(unpacked_q)[q_idx / elements_per_thread] = val_vec;

  } else if (offset < q_hidden + k_hidden) {
    // K: Apply RoPE and write DIRECTLY to K cache
    const int k_offset = offset - q_hidden;
    const int n = k_offset / head_size;
    const int h = k_offset % head_size;
    const int h_idx = h / elements_per_thread;

    if (cos_cache != nullptr && rotary_dim > 0 && h < rotary_dim) {
      // For half-split RoPE, pair values should be read relative to the START of the current K head.
      // Calculate offset to head start: (b, s, k_head_n, 0) in packed QKV.
      const int64_t k_head_start_in_packed = static_cast<int64_t>(b) * sequence_length * d +
                                             static_cast<int64_t>(s) * d +
                                             q_hidden +
                                             static_cast<int64_t>(n) * head_size;
      RotaryDispatcher<LoadT, T>::apply(val_vec,
                                        reinterpret_cast<const LoadT*>(cos_cache),
                                        reinterpret_cast<const LoadT*>(sin_cache),
                                        rotary_dim, h_idx, pos_id, interleaved,
                                        reinterpret_cast<const LoadT*>(packed_qkv),
                                        k_head_start_in_packed / elements_per_thread);
    }

    const int cache_s = past_seq_len + s;
    int64_t cache_idx;
    if (is_cache_bnsh) {
      cache_idx = static_cast<int64_t>(b) * kv_num_heads * max_seqlen * head_size +
                  static_cast<int64_t>(n) * max_seqlen * head_size +
                  static_cast<int64_t>(cache_s) * head_size + h;
    } else {  // BSNH
      cache_idx = static_cast<int64_t>(b) * max_seqlen * kv_num_heads * head_size +
                  static_cast<int64_t>(cache_s) * kv_num_heads * head_size +
                  static_cast<int64_t>(n) * head_size + h;
    }
    // Vector store to k_cache
    reinterpret_cast<LoadT*>(k_cache)[cache_idx / elements_per_thread] = val_vec;

  } else {
    // V: Write DIRECTLY to V cache (no rotation)
    const int v_offset = offset - q_hidden - k_hidden;
    const int n = v_offset / head_size;
    const int h = v_offset % head_size;

    const int cache_s = past_seq_len + s;
    int64_t cache_idx;
    if (is_cache_bnsh) {
      cache_idx = static_cast<int64_t>(b) * kv_num_heads * max_seqlen * head_size +
                  static_cast<int64_t>(n) * max_seqlen * head_size +
                  static_cast<int64_t>(cache_s) * head_size + h;
    } else {  // BSNH
      cache_idx = static_cast<int64_t>(b) * max_seqlen * kv_num_heads * head_size +
                  static_cast<int64_t>(cache_s) * kv_num_heads * head_size +
                  static_cast<int64_t>(n) * head_size + h;
    }
    // Vector store to v_cache
    reinterpret_cast<LoadT*>(v_cache)[cache_idx / elements_per_thread] = val_vec;
  }
}

// Launcher for fused UnpackQKV + RoPE + KV Append
template <typename T>
Status LaunchUnpackQKVWithRoPEAndAppendKV(
    const T* packed_qkv,
    T* unpacked_q,
    T* k_cache,
    T* v_cache,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int sequence_length,
    const int batch_size,
    const int max_seqlen,
    const int* past_seq_lens,
    const T* cos_cache,
    const T* sin_cache,
    const int rotary_dim,
    const int64_t* position_ids,
    const bool interleaved,
    const bool is_cache_bnsh,
    cudaStream_t stream,
    const int max_threads_per_block) {
  // Determine vectorization factor (float4 is 16 bytes)
  constexpr int vector_bytes = sizeof(float4);
  constexpr int element_bytes = sizeof(T);
  constexpr int elements_per_vector = vector_bytes / element_bytes;

  // Validate head_size alignment
  if (head_size % elements_per_vector != 0) {
    // If strict alignment is not met (unlikely given GQA constraints), we should fall back or fail.
    // Typically GQA enforces head_size % 8 == 0.
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by ", elements_per_vector, " for vectorized GQA kernel.");
  }

  // Validate grid dimensions - CUDA limits gridDim.y to 65535
  if (sequence_length > 65535) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Sequence length ", sequence_length,
                           " exceeds CUDA grid dimension limit (65535) for fused UnpackQKV kernel.");
  }

#ifndef NDEBUG
  // Debug-mode alignment assertions for vectorized memory access
  assert(reinterpret_cast<uintptr_t>(packed_qkv) % 16 == 0 && "packed_qkv must be 16-byte aligned");
  assert(reinterpret_cast<uintptr_t>(unpacked_q) % 16 == 0 && "unpacked_q must be 16-byte aligned");
  assert(reinterpret_cast<uintptr_t>(k_cache) % 16 == 0 && "k_cache must be 16-byte aligned");
  assert(reinterpret_cast<uintptr_t>(v_cache) % 16 == 0 && "v_cache must be 16-byte aligned");
  if (cos_cache != nullptr) {
    assert(reinterpret_cast<uintptr_t>(cos_cache) % 16 == 0 && "cos_cache must be 16-byte aligned");
    assert(reinterpret_cast<uintptr_t>(sin_cache) % 16 == 0 && "sin_cache must be 16-byte aligned");
  }
#endif

  // QKV hidden dimension stride
  const int d = (num_heads + 2 * kv_num_heads) * head_size;
  const int d_vectors = d / elements_per_vector;  // Number of vectors per (b, s)

  // 3D grid layout for eliminating integer divisions in kernel:
  //   grid.x = number of blocks needed to cover d_vectors with threads_per_block threads
  //   grid.y = sequence_length
  //   grid.z = batch_size
  const int threads_per_block = std::min(max_threads_per_block, d_vectors);
  const int blocks_x = (d_vectors + threads_per_block - 1) / threads_per_block;
  const dim3 grid(blocks_x, sequence_length, batch_size);
  const dim3 block(threads_per_block);

  UnpackQKVWithRoPEAndAppendKV<T><<<grid, block, 0, stream>>>(
      packed_qkv,
      unpacked_q,
      k_cache,
      v_cache,
      num_heads,
      kv_num_heads,
      head_size,
      d,
      max_seqlen,
      past_seq_lens,
      cos_cache,
      sin_cache,
      rotary_dim,
      position_ids,
      interleaved,
      is_cache_bnsh);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit template instantiations
template Status LaunchUnpackQKVWithRoPEAndAppendKV<half>(
    const half*, half*, half*, half*,
    int, int, int, int, int, int, const int*,
    const half*, const half*, int, const int64_t*, bool, bool,
    cudaStream_t, int);

template Status LaunchUnpackQKVWithRoPEAndAppendKV<BFloat16>(
    const BFloat16*, BFloat16*, BFloat16*, BFloat16*,
    int, int, int, int, int, int, const int*,
    const BFloat16*, const BFloat16*, int, const int64_t*, bool, bool,
    cudaStream_t, int);

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

////////// Kernels (supports right padding but not left padding)

#if USE_FLASH_ATTENTION

// Use flash attention for all workloads (rotary, kv append, attention, etc.). No extra kernel is used in this path.
// Currently, only decoding or subsequent prompt can use this path. First prompt will not use this path.
template <typename T>
Status FlashAttentionDecoding(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data,
    float scale) {
  assert(!parameters.is_first_prompt && parameters.kv_share_buffer);

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  bool is_causal = parameters.is_unidirectional;
  bool is_bf16 = std::is_same<T, BFloat16>::value;

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

  void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
  void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));
  void* cos_cache = reinterpret_cast<void*>(const_cast<T*>(data.cos_cache));
  void* sin_cache = reinterpret_cast<void*>(const_cast<T*>(data.sin_cache));
  void* head_sink = reinterpret_cast<void*>(const_cast<T*>(data.head_sink));

  bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;

  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
      device_prop, stream, query, present_key, present_value, key, value, data.output,
      reinterpret_cast<void*>(data.softmax_lse), seqlens_k, cos_cache, sin_cache,
      /*cache_batch_idx*/ nullptr, /*leftpad_k*/ nullptr, head_sink, /*block_table*/ nullptr,
      batch_size, num_heads, kv_num_heads, head_size, sequence_length,
      parameters.seqlen_present_kv_cache, kv_sequence_length, parameters.rotary_dim,
      scale, parameters.softcap, is_causal, is_bf16, parameters.use_smooth_softmax, past_bsnh, parameters.num_splits,
      reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum),
      parameters.local_window_size - 1, parameters.rotary_interleaved, parameters.is_packed_qkv));

  return Status::OK();
}

// Use extra kernel(s) for unpacking, rotary and kv append.
// Flash attention is used for attention only.
template <typename T>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data,
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
  bool is_bf16 = std::is_same<T, BFloat16>::value;

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

#if DUMP_TENSOR_LEVEL > 0
  printf("[GQA FlashAttention] is_packed_qkv: %d, is_first_prompt: %d, is_subsequent_prompt: %d, kv_share_buffer: %d\n",
         static_cast<int>(parameters.is_packed_qkv),
         static_cast<int>(parameters.is_first_prompt),
         static_cast<int>(parameters.is_subsequent_prompt),
         static_cast<int>(parameters.kv_share_buffer));
#endif
  DUMP_TENSOR_INIT();

  // Track whether we keep packed QKV for FA kernels
  bool use_packed_for_fa = parameters.is_packed_qkv;

  // Track if we used the fully fused path (packed + share_buffer + rotary)
  bool used_fused_packed_path = false;

  // =========================================================================
  // Handle Packed QKV Input
  // =========================================================================
  if (parameters.is_packed_qkv) {
    T* unpacked_buffer = reinterpret_cast<T*>(data.unpacked_qkv_buffer);
    if (unpacked_buffer != nullptr) {
      T* unpacked_q = unpacked_buffer;

      // Check if we can use the fully fused path
      if (parameters.kv_share_buffer && parameters.do_rotary && !data.disable_fused_kv) {
        // FULLY FUSED PATH: Unpack + RoPE Q + RoPE K + Append KV in single kernel
        // This eliminates 4 kernel launches!
        ORT_RETURN_IF_ERROR(LaunchUnpackQKVWithRoPEAndAppendKV<T>(
            reinterpret_cast<const T*>(data.query),  // packed QKV
            unpacked_q,                              // Q output buffer (rotated)
            data.present_key,                        // K cache (direct write)
            data.present_value,                      // V cache (direct write)
            num_heads,
            kv_num_heads,
            head_size,
            sequence_length,
            batch_size,
            parameters.seqlen_present_kv_cache,
            data.past_seq_lens,
            data.cos_cache,
            data.sin_cache,
            parameters.rotary_dim,
            data.position_ids,
            parameters.rotary_interleaved,
            !past_bsnh,  // is_cache_bnsh
            stream,
            max_threads_per_block));

        // Update query to point to rotated Q
        query = unpacked_q;
        use_packed_for_fa = false;
        used_fused_packed_path = true;

        // Track buffer usage: Only Q is stored in unpacked_qkv_buffer (fused path writes K/V to cache)
        size_t q_bytes = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size * sizeof(T);
        UpdateUnpackedQkvMaxUsed(data, q_bytes);

        // K and V are already in cache - no need to set key/value pointers

      } else {
        // Standard path: Unpack first, then process K/V separately
        size_t q_size = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size;
        T* unpacked_k = unpacked_buffer + q_size;

        size_t k_size = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size;
        T* unpacked_v = unpacked_k + k_size;

        // If we need Q rotation, we MUST unpack Q as well.
        T* q_dst = parameters.do_rotary ? unpacked_q : nullptr;

        // Always unpack to BSNH as LaunchConcatNewToPastKV expects contiguous BSNH input
        ORT_RETURN_IF_ERROR((LaunchUnpackQKV<T, false>(reinterpret_cast<const T*>(data.query), q_dst, unpacked_k, unpacked_v, num_heads, kv_num_heads, head_size, sequence_length, batch_size, stream, max_threads_per_block)));

        // Update key/value to point to unpacked buffers
        key = unpacked_k;
        value = unpacked_v;

        if (parameters.do_rotary) {
          query = unpacked_q;
          use_packed_for_fa = false;
        }

        // Track buffer usage: Q+K+V unpacked
        size_t total_bytes = (q_size + 2 * k_size) * sizeof(T);
        UpdateUnpackedQkvMaxUsed(data, total_bytes);
      }
    }
  }
  // =========================================================================
  // Handle Unpacked Q, K, V Input (with optional RoPE)
  // =========================================================================
  else {
    if (parameters.do_rotary) {
      // For unpacked input, we need to rotate Q and K.
      // The rotated Q and K will be stored in unpacked_qkv_buffer with layout [Q (B*S*H*D), K (B*S*H_kv*D)].
      T* unpacked_buffer = reinterpret_cast<T*>(data.unpacked_qkv_buffer);
      if (unpacked_buffer != nullptr) {
        query = unpacked_buffer;
        // Do not update key here for Unpacked path.
        // key must remain pointing to data.key (Input) for Explicit K Rotation (k_src).
        // k_dst will be calculated from unpacked_buffer explicitly.
      }
    }
  }

  const int64_t* position_ids = data.position_ids;

  // Explicit Q Rotation (skip if fused path already applied RoPE)
  if (parameters.do_rotary && !used_fused_packed_path) {
    // Rotate Q
    // Q ptr is already set to the destination buffer (unpacked_buffer) above.
    // Input for Rotation:
    //   If packed: we unpacked into `query` buffer. So Input==Output (In-place).
    //   If unpacked: we set `query = unpacked_buffer`. But Input is `data.query`.
    const T* q_input_for_rope = parameters.is_packed_qkv ? reinterpret_cast<const T*>(query) : reinterpret_cast<const T*>(data.query);
    T* q_output_for_rope = reinterpret_cast<T*>(query);  // Destination

    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel(
        stream,
        q_output_for_rope,
        q_input_for_rope,
        nullptr,  // position_ids unused for format 2/3
        data.past_seq_lens,
        data.cos_cache,
        data.sin_cache,
        batch_size,
        sequence_length,
        num_heads,
        head_size,
        parameters.rotary_dim,
        parameters.max_sequence_length,
        2,  // position_ids_format = 2 (Implicit: past_seq_lens[b] + s)
        parameters.rotary_interleaved,
        max_threads_per_block,
        false  // is_input_bnsh_format (Q is BSNH)
        ));
    DUMP_TENSOR("Rotated Q", q_output_for_rope, batch_size, sequence_length, num_heads, head_size);

    // Rotate K will be done later in fused kernel.
  }

  // Skip KV append if we used the fully fused path (KV already in cache)
  if (!used_fused_packed_path) {
    if (parameters.kv_share_buffer && !parameters.is_first_prompt) {
      constexpr bool is_new_kv_bnsh_format = false;
      if (parameters.do_rotary) {
        // Explicit K Rotation (replacing internal RoPE in fused kernel)
        size_t q_elements = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size;
        T* k_dst = reinterpret_cast<T*>(data.unpacked_qkv_buffer) + q_elements;
        const T* k_src = reinterpret_cast<const T*>(key);

        ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel(
            stream,
            k_dst,
            k_src,
            position_ids,
            data.past_seq_lens,
            data.cos_cache,
            data.sin_cache,
            batch_size,
            sequence_length,
            kv_num_heads,
            head_size,
            parameters.rotary_dim,
            parameters.max_sequence_length,
            position_ids != nullptr ? 1 : 2,
            parameters.rotary_interleaved,
            max_threads_per_block,
            false));

        if (!data.disable_fused_kv) {
          // Use fused kernel for K (rotated) + V append
          ORT_RETURN_IF_ERROR(LaunchConcatKVInPlaceFused<T>(
              batch_size,
              kv_num_heads,
              head_size,
              parameters.seqlen_present_kv_cache,
              data.past_seq_lens,
              data.total_seq_lens,
              sequence_length,
              k_dst,
              reinterpret_cast<const T*>(data.value),
              data.present_key,
              data.present_value,
              !past_bsnh,
              is_new_kv_bnsh_format,
              stream,
              max_threads_per_block));
        } else {
          // Unfused Fallback: LaunchConcatKVInPlace
          // We must pass the ROTATED K (k_dst) to it.
          ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(
              parameters, data, k_dst, value, is_new_kv_bnsh_format, stream, max_threads_per_block));
        }

        // Track buffer usage: Q + K rotated in unpacked_qkv_buffer
        size_t k_elements = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size;
        size_t total_bytes = (q_elements + k_elements) * sizeof(T);
        UpdateUnpackedQkvMaxUsed(data, total_bytes);
      } else {
        // No RoPE - use original kernel
        ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(parameters, data, key, value, is_new_kv_bnsh_format, stream, max_threads_per_block));
      }
    } else {
      // ORT MUST perform the append (using unpacked data for packed case)
      bool skip_new_append = false;
      // FUSED ROPE: Pass RoPE params to ConcatKV (applies RoPE to K as it is appended)
      // IMPORTANT: For Fused RoPE with unpacked input, we must pass data.key (the original input),
      // not the scratch buffer 'key' which is empty since explicit rotation was skipped.
      const void* key_for_concat = parameters.is_packed_qkv ? key : data.key;
      ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKVHelper<T>(parameters, data, key_for_concat, value, stream, max_threads_per_block, skip_new_append,
                                                           data.cos_cache, data.sin_cache, parameters.rotary_dim, nullptr, parameters.rotary_interleaved));
    }
  }

  DUMP_TENSOR("Total Seq Lens", data.total_seq_lens, batch_size, 1);
  DUMP_TENSOR("Past Seq Lens", data.past_seq_lens, batch_size, 1);
  DUMP_TENSOR("Present Key", data.present_key, batch_size, parameters.seqlen_present_kv_cache, kv_num_heads, head_size);
  DUMP_TENSOR("Present Value", data.present_value, batch_size, parameters.seqlen_present_kv_cache, kv_num_heads, head_size);

  void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
  void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

  // Disable internal RoPE in Flash Attention (pass nullptr)
  void* cos_cache = nullptr;
  void* sin_cache = nullptr;
  void* head_sink = reinterpret_cast<void*>(const_cast<T*>(data.head_sink));

  // We have already appended (and quantized if needed) the new tokens into present_key/value.
  // Pass nullptr for new_k/new_v to disable flash attention kernel's internal Append_KV logic.
  void* kernel_new_k = nullptr;
  void* kernel_new_v = nullptr;

  // Use padded seq lens for first prompt since mha_fwd_kvcache assumes uniform seqlen_q.
  // The causal mask offset (seqlen_k - seqlen_q) becomes negative when seqlen_k < seqlen_q, causing incorrect masking.
  int* seq_lens = parameters.is_first_prompt ? data.padded_seq_lens : data.total_seq_lens;

  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
      device_prop, stream, query, present_key, present_value,
      kernel_new_k, kernel_new_v,
      data.output, reinterpret_cast<void*>(data.softmax_lse), seq_lens, cos_cache, sin_cache,
      /*cache_batch_idx*/ nullptr, /*leftpad_k*/ nullptr, head_sink, /*block_table*/ nullptr,
      batch_size, num_heads, kv_num_heads, head_size, sequence_length,
      parameters.seqlen_present_kv_cache, kv_sequence_length,
      parameters.rotary_dim, scale, parameters.softcap, is_causal, is_bf16,
      parameters.use_smooth_softmax, past_bsnh, parameters.num_splits,
      reinterpret_cast<void*>(data.softmax_lse_accum),
      reinterpret_cast<void*>(data.out_accum), parameters.local_window_size - 1,
      parameters.rotary_interleaved, use_packed_for_fa, 0, 1));

  return Status::OK();
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
template <typename T>
Status EfficientAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int present_sequence_length = parameters.seqlen_present_kv_cache;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

#if DUMP_TENSOR_LEVEL > 0
  printf("[GQA EfficientAttention] is_packed_qkv: %d, is_first_prompt: %d, is_subsequent_prompt: %d, kv_share_buffer: %d\n",
         static_cast<int>(parameters.is_packed_qkv),
         static_cast<int>(parameters.is_first_prompt),
         static_cast<int>(parameters.is_subsequent_prompt),
         static_cast<int>(parameters.kv_share_buffer));
#endif

  const void* query;
  const void* key;
  const void* value;

  if (!parameters.is_packed_qkv) {
    query = reinterpret_cast<const void*>(data.query);
    key = reinterpret_cast<const void*>(data.key);
    value = reinterpret_cast<const void*>(data.value);
  } else {
    size_t q_size = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size;
    size_t k_size = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size;
    auto q = reinterpret_cast<T*>(data.unpacked_qkv_buffer);
    auto k = reinterpret_cast<T*>(data.unpacked_qkv_buffer + q_size);
    auto v = reinterpret_cast<T*>(data.unpacked_qkv_buffer + q_size + k_size);

    Status status = LaunchUnpackQKV<T, LAYOUT_BSNH>(
        reinterpret_cast<const T*>(data.query), q, k, v, num_heads, kv_num_heads,
        head_size, sequence_length, batch_size, stream, max_threads_per_block);
    if (status != Status::OK()) {
      return status;
    }

    query = reinterpret_cast<const void*>(q);
    key = reinterpret_cast<const void*>(k);
    value = reinterpret_cast<const void*>(v);

    // Track buffer usage: Q+K+V unpacked
    size_t total_bytes = (q_size + 2 * k_size) * sizeof(T);
    UpdateUnpackedQkvMaxUsed(data, total_bytes);
  }

  const int64_t* position_ids = data.position_ids;
  if (parameters.do_rotary) {
    auto q_buffer = reinterpret_cast<T*>(data.rotary_buffer);

    // Launch rotary embedding kernel for Q
    if (position_ids != nullptr) {
      // User provided explicit position_ids - Use Format 1
      ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
          stream, q_buffer, reinterpret_cast<const T*>(query),
          position_ids, nullptr /*past_seq_lens not used in format 1*/,
          data.cos_cache, data.sin_cache,
          parameters.batch_size, parameters.sequence_length,
          parameters.num_heads, parameters.head_size,
          parameters.rotary_dim, parameters.max_sequence_length,
          1,  // Format 1: Explicit position_ids
          parameters.rotary_interleaved,
          max_threads_per_block,
          false));
    } else {
      ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
          stream, q_buffer, reinterpret_cast<const T*>(query),
          nullptr, data.past_seq_lens,
          data.cos_cache, data.sin_cache,
          parameters.batch_size, parameters.sequence_length,
          parameters.num_heads, parameters.head_size,
          parameters.rotary_dim, parameters.max_sequence_length,
          2,  // Format 2: Implicit (past_seq_lens[b] + s)
          parameters.rotary_interleaved,
          max_threads_per_block,
          false));
    }
    query = reinterpret_cast<const void*>(q_buffer);

    // For kv_share_buffer path, we use Fused RoPE in LaunchConcatKVInPlaceWithRoPE.
    // For non-share-buffer path, we use Fused RoPE in LaunchConcatNewToPastKVHelper.
    // No explicit K rotation needed here - handled by fused kernels.

    // key remains pointing to original source for use in fused kernel below

    // Track rotary buffer usage: Q rotated (K rotation is fused in KV append)
    size_t q_bytes = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size * sizeof(T);
    size_t k_bytes = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size * sizeof(T);
    // Note: rotary_buffer layout is [Q_rotated, K_rotated] - no position_ids here
    UpdateRotaryMaxUsed(data, q_bytes + k_bytes);

    // Track position_ids_buffer usage
    size_t pos_ids_bytes = static_cast<size_t>(batch_size) * sequence_length * sizeof(int64_t);
    UpdatePositionIdsMaxUsed(data, pos_ids_bytes);
  }

  if (parameters.kv_share_buffer) {
    // Concatenate new kv in place
    constexpr bool is_new_kv_bnsh_format = false;

    if (parameters.do_rotary) {
      // Explicit K Rotation
      size_t q_elements = static_cast<size_t>(batch_size) * sequence_length * num_heads * head_size;
      T* k_dst = reinterpret_cast<T*>(data.rotary_buffer) + q_elements;
      const T* k_src = reinterpret_cast<const T*>(key);

      ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel(
          stream,
          k_dst,
          k_src,
          position_ids,
          data.past_seq_lens,
          data.cos_cache,
          data.sin_cache,
          batch_size,
          sequence_length,
          parameters.kv_num_heads,
          parameters.head_size,
          parameters.rotary_dim,
          parameters.max_sequence_length,
          position_ids != nullptr ? 1 : 2,
          parameters.rotary_interleaved,
          max_threads_per_block,
          false));

      if (!data.disable_fused_kv) {
        // Use truly fused kernel for K (already rotated) + V append in single kernel
        ORT_RETURN_IF_ERROR(LaunchConcatKVInPlaceFused<T>(
            batch_size,
            parameters.kv_num_heads,
            parameters.head_size,
            parameters.seqlen_present_kv_cache,
            data.past_seq_lens,
            data.total_seq_lens,
            parameters.sequence_length,
            k_dst,
            reinterpret_cast<const T*>(value),
            data.present_key,
            data.present_value,
            past_kv_format != AttentionQkvFormat::Q_K_V_BSNH,  // is_past_kv_bnsh_format
            is_new_kv_bnsh_format,
            stream,
            max_threads_per_block));
      } else {
        // Unfused Fallback
        ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(
            parameters, data, k_dst, value, is_new_kv_bnsh_format, stream, max_threads_per_block));
      }

      // Track rotary buffer usage: Q + K rotated (no position_ids in rotary_buffer)
      size_t k_elements = static_cast<size_t>(batch_size) * sequence_length * kv_num_heads * head_size;
      UpdateRotaryMaxUsed(data, (q_elements + k_elements) * sizeof(T));
    } else {
      // No RoPE - use original kernel
      ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(
          parameters, data, key, value, is_new_kv_bnsh_format, stream, max_threads_per_block));
    }
  } else {
    // Copy past and concat new KV to present buffer
    // FUSED ROPE: Pass RoPE params to ConcatKV
    ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKVHelper<T>(parameters, data, key, value, stream, max_threads_per_block, false,
                                                         data.cos_cache, data.sin_cache, parameters.rotary_dim, nullptr, parameters.rotary_interleaved));
  }

  // Ungroup if grouped, otherwise use present kv directly
  const bool is_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
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
                                         present_sequence_length, is_bsnh, stream, max_threads_per_block));
    key = reinterpret_cast<const void*>(data.k);
    value = reinterpret_cast<const void*>(data.v);
  }

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_bf16 = std::is_same<T, BFloat16>::value;
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
  p.is_kv_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
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

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& /*cublas*/,
    Stream* ort_stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention_fast_decode) {
    return FlashAttentionDecoding(device_prop, stream, parameters, data, scale);
  }

  if (data.use_flash_attention) {
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

template struct GroupQueryAttentionData<half>;

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<half>& data);

template Status LaunchUnpackQKV<half, LAYOUT_BNSH>(
    const half* packed_qkv, half* unpacked_q, half* unpacked_k, half* unpacked_v, const int num_heads,
    const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size,
    cudaStream_t stream, const int max_threads_per_block);

template struct GroupQueryAttentionData<BFloat16>;

template Status QkvToContext<BFloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<BFloat16>& data);

template Status LaunchUnpackQKV<BFloat16, LAYOUT_BNSH>(
    const BFloat16* packed_qkv, BFloat16* unpacked_q, BFloat16* unpacked_k, BFloat16* unpacked_v, const int num_heads,
    const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size,
    cudaStream_t stream, const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
