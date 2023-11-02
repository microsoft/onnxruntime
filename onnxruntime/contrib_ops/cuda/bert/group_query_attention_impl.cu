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
// (1) support GPT-2 past state, unidirectional mask (causal)
// (2) use flash attention kernel from (https://github.com/Dao-AILab/flash-attention)
// (3) support different number of heads for Q and KV
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

////////// Auxiliary Kernels for KV prep

// Kernel for seqlens_k
__global__ void repeat_seqlen(int32_t* seqlens_k, int32_t seqlen, int batch_size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < batch_size) seqlens_k[id] = seqlen;
}

// Kernel to append new and past kv in either BSNH or BNSH format
// Adapted from ConcatTensorToTensor kernel in attention_kv_cache.cu file
template <typename T>
__global__ void ConcatNewToPastKV(const int new_seqlen,
                                  const T* past_kv,
                                  const T* new_kv,
                                  T* present_kv,
                                  const bool is_bsnh) {  // refers to past; otherwise bnsh
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int present_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int present_batch_stride = present_seqlen * num_heads * H;
  const int row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : present_seqlen * H;

  // past_kv:     BPNH or BNPH
  // new_kv:      BLNH
  // present_kv:  BTNH or BNTH, where T = P + L
  const int past_seqlen = present_seqlen - new_seqlen;

  int out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
  if (s < past_seqlen) {
    const int past_batch_stride = past_seqlen * num_heads * H;
    const int past_head_stride = is_bsnh ? H : past_seqlen * H;
    const int in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
    present_kv[out_offset] = past_kv[in_offset];
  } else if (s < present_seqlen) {
    // Note: new KV always BSNH
    const int new_batch_stride = new_seqlen * num_heads * H;
    const int new_row_stride = num_heads * H;
    const int new_head_stride = H;
    const int in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;
    present_kv[out_offset] = new_kv[in_offset];
  }
}

// Use when (H*)*num_heads > 1024
template <typename T>
__global__ void ConcatNewToPastKVLarge(const int new_seqlen,
                                       const int H,
                                       const int num_heads,
                                       const T* past_kv,
                                       const T* new_kv,
                                       T* present_kv,
                                       const bool is_bsnh) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int present_seqlen = gridDim.y;

    const int present_batch_stride = present_seqlen * num_heads * H;
    const int row_stride = is_bsnh ? num_heads * H : H;
    const int present_head_stride = is_bsnh ? H : present_seqlen * H;

    // past_kv:     BPNH or BNPH
    // new_kv:      BLNH
    // present_kv:  BTNH or BNTH, where T = P + L
    const int past_seqlen = present_seqlen - new_seqlen;

    int out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
    if (s < past_seqlen) {
      const int past_batch_stride = past_seqlen * num_heads * H;
      const int past_head_stride = is_bsnh ? H : past_seqlen * H;
      const int in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
      present_kv[out_offset] = past_kv[in_offset];
    } else if (s < present_seqlen) {
      const int new_batch_stride = new_seqlen * num_heads * H;
      const int new_row_stride = num_heads * H;
      const int new_head_stride = H;
      const int in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;
      present_kv[out_offset] = new_kv[in_offset];
    }
  }
}

// Concat new to past in present. Supports past BSNH or past BNSH
template <typename T>
Status LaunchConcatNewToPastKV(contrib::GroupQueryAttentionParameters& parameters,
                               GroupQueryAttentionData<T>& data,
                               cudaStream_t stream,
                               const int max_threads_per_block) {
  const int batch_size = parameters.batch_size;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int present_sequence_length = parameters.present_sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  const int H = head_size / 4;  // divide by 4 so kernel can operate on 4 float16 elements at a time.
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(present_sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatNewToPastKV<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                          reinterpret_cast<const float2*>(data.past_key),
                                                          reinterpret_cast<const float2*>(data.key),
                                                          reinterpret_cast<float2*>(data.present_key),
                                                          past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatNewToPastKV<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                          reinterpret_cast<const float2*>(data.past_value),
                                                          reinterpret_cast<const float2*>(data.value),
                                                          reinterpret_cast<float2*>(data.present_value),
                                                          past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  } else {
    int steps = (H * kv_num_heads + 255) / 256;
    const dim3 grid(steps, present_sequence_length, batch_size);
    const dim3 block(256, 1, 1);
    ConcatNewToPastKVLarge<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                               H,
                                                               kv_num_heads,
                                                               reinterpret_cast<const float2*>(data.past_key),
                                                               reinterpret_cast<const float2*>(data.key),
                                                               reinterpret_cast<float2*>(data.present_key),
                                                               past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatNewToPastKVLarge<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                               H,
                                                               kv_num_heads,
                                                               reinterpret_cast<const float2*>(data.past_value),
                                                               reinterpret_cast<const float2*>(data.value),
                                                               reinterpret_cast<float2*>(data.present_value),
                                                               past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  }
  return CUDA_CALL(cudaGetLastError());
}

// Kernel to append new kv to kv buffer in place
template <typename T>
__global__ void ConcatKVInPlace(const int past_seqlen,
                                const int present_seqlen,
                                T* kv_buff,
                                const T* new_kv,
                                const bool is_bsnh) {  // refers to kv buff; otherwise bnsh
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int present_batch_stride = present_seqlen * num_heads * H;
  const int present_row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : present_seqlen * H;

  // kv_buff:     BTNH or BNTH with buffered memory for new
  // new_kv:      BLNH

  int out_offset = b * present_batch_stride + (s + past_seqlen) * present_row_stride + n * present_head_stride + h;
  // Note: new KV always BSNH
  const int new_batch_stride = new_seqlen * num_heads * H;
  const int new_row_stride = num_heads * H;
  const int new_head_stride = H;
  const int in_offset = b * new_batch_stride + s * new_row_stride + n * new_head_stride + h;
  kv_buff[out_offset] = new_kv[in_offset];
}

template <typename T>
__global__ void ConcatKVInPlaceLarge(const int past_seqlen,
                                     const int present_seqlen,
                                     const int H,
                                     const int num_heads,
                                     T* kv_buff,
                                     const T* new_kv,
                                     const bool is_bsnh) {  // refers to kv buff; otherwise bnsh
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int new_seqlen = gridDim.y;

    const int present_batch_stride = present_seqlen * num_heads * H;
    const int present_row_stride = is_bsnh ? num_heads * H : H;
    const int present_head_stride = is_bsnh ? H : present_seqlen * H;

    // kv_buff:     BTNH or BNTH with buffered memory for new
    // new_kv:      BLNH

    int out_offset = b * present_batch_stride + (s + past_seqlen) * present_row_stride + n * present_head_stride + h;
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
Status LaunchConcatKVInPlace(contrib::GroupQueryAttentionParameters& parameters,
                             GroupQueryAttentionData<T>& data,
                             cudaStream_t stream,
                             const int max_threads_per_block) {
  const int batch_size = parameters.batch_size;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int present_sequence_length = parameters.present_sequence_length;
  const int past_sequence_length = parameters.past_sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;
  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  const int H = head_size / 4;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(kv_sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(past_sequence_length,
                                                        present_sequence_length,
                                                        reinterpret_cast<float2*>(data.present_key),
                                                        reinterpret_cast<const float2*>(data.key),
                                                        past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(past_sequence_length,
                                                        present_sequence_length,
                                                        reinterpret_cast<float2*>(data.present_value),
                                                        reinterpret_cast<const float2*>(data.value),
                                                        past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  } else {
    int steps = int(ceil(float(H * kv_num_heads) / 256.0));
    const dim3 grid(steps, kv_sequence_length, batch_size);
    const dim3 block(256, 1, 1);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(past_sequence_length,
                                                             present_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(data.present_key),
                                                             reinterpret_cast<const float2*>(data.key),
                                                             past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(past_sequence_length,
                                                             present_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(data.present_value),
                                                             reinterpret_cast<const float2*>(data.value),
                                                             past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  }
  return CUDA_CALL(cudaGetLastError());
}

// Kernel for use with memory efficient kernel... kv_in is grouped and of bnsh or bsnh... kv_out is ungrouped and bsnh
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
Status LaunchUngroup(contrib::GroupQueryAttentionParameters& parameters,
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

////////// Launch Kernels

#if USE_FLASH_ATTENTION
template <typename T>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int present_sequence_length = parameters.present_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
  void* key = reinterpret_cast<void*>(const_cast<T*>(data.key));
  void* value = reinterpret_cast<void*>(const_cast<T*>(data.value));

  bool is_causal = parameters.is_unidirectional;

  if (data.past_key != nullptr && data.past_key == data.present_key) {
    // Share buffer case
    void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
    void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

    // Launch kernel to copy seqlen
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(batch_size) / thr_per_blk);
    repeat_seqlen<<<blk_in_grid, thr_per_blk, 0, stream>>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

    bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, stream, query, present_key, present_value, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
        reinterpret_cast<void*>(data.seqlens_k), batch_size, num_heads, kv_num_heads,
        head_size, sequence_length, present_sequence_length, kv_sequence_length,
        scale, is_causal, past_bsnh, parameters.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum),
        reinterpret_cast<void*>(data.out_accum)));

  } else {
    // Not share buffer or no past (prompt generation)
    // Note that Flash Attention kv-caching operates in place on a buffer... therefore this path is inneficient
    ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKV(parameters, data, stream, max_threads_per_block));

    void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
    void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

    bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
        device_prop, stream, query, present_key, present_value, data.output, reinterpret_cast<void*>(data.softmax_lse),
        batch_size, num_heads, kv_num_heads, head_size,
        sequence_length, present_sequence_length, scale, is_causal, parameters.num_splits,
        reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum), past_bsnh));
  }

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, num_heads, head_size);

  return Status::OK();
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
template <typename T>
Status EfficientAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int past_sequence_length = parameters.past_sequence_length;
  const int present_sequence_length = parameters.present_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  const void* query = reinterpret_cast<const void*>(data.query);
  const void* key = reinterpret_cast<const void*>(data.key);
  const void* value = reinterpret_cast<const void*>(data.value);
  if (data.past_key != nullptr) {
    // Past key case
    // concatenate new kv to past kv
    if (data.past_key == data.present_key) {
      ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(parameters, data, stream, max_threads_per_block));
    } else {
      ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKV(parameters, data, stream, max_threads_per_block));
    }
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
      ORT_RETURN_IF_ERROR(LaunchUngroup(parameters, k_buff, v_buff, k_og, v_og, past_sequence_length + kv_sequence_length,
                                        present_sequence_length, is_bsnh, stream, max_threads_per_block));
      key = reinterpret_cast<const void*>(data.k);
      value = reinterpret_cast<const void*>(data.v);
    }
  } else if (num_heads == kv_num_heads) {
    // no past or present and no need to ungroup... still copy kv into present buffer
    ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKV(parameters, data, stream, max_threads_per_block));
    key = reinterpret_cast<const void*>(data.present_key);
    value = reinterpret_cast<const void*>(data.present_value);
  } else {
    // intermediate buffer so q and kv have same num heads... still copy kv into present buffer
    ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKV(parameters, data, stream, max_threads_per_block));
    float2* k_buff = reinterpret_cast<float2*>(data.k);
    float2* v_buff = reinterpret_cast<float2*>(data.v);
    const float2* k_og = reinterpret_cast<const float2*>(data.present_key);
    const float2* v_og = reinterpret_cast<const float2*>(data.present_value);
    ORT_RETURN_IF_ERROR(LaunchUngroup(parameters, k_buff, v_buff, k_og, v_og, kv_sequence_length,
                                      kv_sequence_length, past_kv_format == AttentionQkvFormat::Q_K_V_BSNH, stream,
                                      max_threads_per_block));
    key = reinterpret_cast<const void*>(data.k);
    value = reinterpret_cast<const void*>(data.v);
  }

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_half = sizeof(T) == 2;
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.sequence_length = sequence_length;
  p.kv_sequence_length = past_sequence_length + kv_sequence_length;
  p.max_sequence_length = (num_heads == kv_num_heads) ? present_sequence_length : past_sequence_length + kv_sequence_length;
  p.qk_head_size = head_size;
  p.v_head_size = head_size;
  p.causal = parameters.is_unidirectional;
  p.scale = scale;
  p.seqlen_k_ptr = nullptr;
  p.seqstart_q_ptr = nullptr;
  p.seqstart_k_ptr = nullptr;
  p.query = query;
  p.key = key;
  p.value = value;
  p.attn_bias = nullptr;
  p.is_attn_bias_batched = false;
  p.is_kv_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
  p.output = data.output;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(p.v_head_size, sizeof(T) == sizeof(float))
                    ? data.fmha_buffer
                    : nullptr;
  p.stream = stream;
  run_memory_efficient_attention(p);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("efficient attention output", data.output, batch_size, sequence_length, num_heads, head_size);

  return Status::OK();
}
#endif

////////// API Functions

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;

#if USE_FLASH_ATTENTION
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
