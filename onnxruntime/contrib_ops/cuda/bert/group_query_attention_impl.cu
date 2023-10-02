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
  if(id < batch_size) seqlens_k[id] = seqlen;
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
  // new_kv:      BLNH or BNLH
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

template <typename T>
__global__ void ConcatNewToPastKVLarge(const int new_seqlen,
                                       const int H,
                                       const T* past_kv,
                                       const T* new_kv,
                                       T* present_kv,
                                       const bool is_bsnh) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int present_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int thread_stride = blockDim.x;

  const int present_batch_stride = present_seqlen * num_heads * H;
  const int row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : present_seqlen * H;

  // past_kv:     BPNH or BNPH
  // new_kv:      BLNH or BNLH
  // present_kv:  BTNH or BNTH, where T = P + L
  const int past_seqlen = present_seqlen - new_seqlen;

  while (h < H) {
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

    h += thread_stride;
  }
}

// Concat new to past in present. Supports past BSNH or past BNSH
template <typename T>
Status LaunchConcatNewToPastKV(contrib::GroupQueryAttentionParameters& parameters,
                               GroupQueryAttentionData<T>& data) {

  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int present_sequence_length = parameters.present_sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  // Note that Flash Attention kv-caching operates in place on a buffer... therefore this path is inneficient
  const int H = head_size / 4;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(present_sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatNewToPastKV<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                            reinterpret_cast<const float2*>(data.past_key),
                                                            reinterpret_cast<const float2*>(data.key),
                                                            reinterpret_cast<float2*>(data.present_key),
                                                            past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatNewToPastKV<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                            reinterpret_cast<const float2*>(data.past_value),
                                                            reinterpret_cast<const float2*>(data.value),
                                                            reinterpret_cast<float2*>(data.present_value),
                                                            past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  } else {
    const dim3 grid(present_sequence_length, batch_size, 1);
    const dim3 block(max_threads_per_block / kv_num_heads, kv_num_heads, 1);
    ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                                  H,
                                                                  reinterpret_cast<const float2*>(data.past_key),
                                                                  reinterpret_cast<const float2*>(data.key),
                                                                  reinterpret_cast<float2*>(data.present_key),
                                                                  past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                                  H,
                                                                  reinterpret_cast<const float2*>(data.past_value),
                                                                  reinterpret_cast<const float2*>(data.value),
                                                                  reinterpret_cast<float2*>(data.present_value),
                                                                  past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  }
  return CUDA_CALL(cudaGetLastError());
}

// Kernel to append new and past kv in either BSNH or BNSH format
// Adapted from ConcatTensorToTensor kernel in attention_kv_cache.cu file
template <typename T>
__global__ void ConcatKVInPlace(const int past_seqlen,
                                const T* kv_buff,
                                const T* new_kv,
                                const bool is_bsnh) {  // refers to past; otherwise bnsh
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int present_seqlen = past_seqlen + new_seqlen;
  const int present_batch_stride = present_seqlen * num_heads * H;
  const int row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : present_seqlen * H;

  // kv_buff:     BTNH or BNTH with buffered memory for new
  // new_kv:      BLNH or BNLH

  int out_offset = b * present_batch_stride + (s + past_seqlen) * row_stride + n * present_head_stride + h;
  // Note: new KV always BSNH
  const int new_batch_stride = new_seqlen * num_heads * H;
  const int new_row_stride = num_heads * H;
  const int new_head_stride = H;
  const int in_offset = b * new_batch_stride + s * new_row_stride + n * new_head_stride + h;
  kv_buff[out_offset] = new_kv[in_offset];
}

// Concat new to past in present. Supports past BSNH or past BNSH
template <typename T>
Status LaunchConcatKVInPlace(contrib::GroupQueryAttentionParameters& parameters,
                               GroupQueryAttentionData<T>& data) {

  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int past_sequence_length = parameters.past_sequence_length;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  // Note that Flash Attention kv-caching operates in place on a buffer... therefore this path is inneficient
  const int H = head_size / 4;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(kv_sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatKVInPlace<float2><<< grid, block, 0, stream >>>(past_sequence_length,
                                                          reinterpret_cast<float2*>(data.present_key),
                                                          reinterpret_cast<const float2*>(data.key),
                                                          past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    ConcatKVInPlace<float2><<< grid, block, 0, stream >>>(past_sequence_length,
                                                          reinterpret_cast<float2*>(data.present_value),
                                                          reinterpret_cast<const float2*>(data.value),
                                                          past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  }
  // TODO(aciddelgado): big gulp version
  // } else {
  //   const dim3 grid(present_sequence_length, batch_size, 1);
  //   const dim3 block(max_threads_per_block / kv_num_heads, kv_num_heads, 1);
  //   ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
  //                                                                 H,
  //                                                                 reinterpret_cast<const float2*>(data.past_key),
  //                                                                 reinterpret_cast<const float2*>(data.key),
  //                                                                 reinterpret_cast<float2*>(data.present_key),
  //                                                                 past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  //   ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
  //                                                                 H,
  //                                                                 reinterpret_cast<const float2*>(data.past_value),
  //                                                                 reinterpret_cast<const float2*>(data.value),
  //                                                                 reinterpret_cast<float2*>(data.present_value),
  //                                                                 past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  // }
  return CUDA_CALL(cudaGetLastError());
}

// Kernel for use with memory efficient kernel... kv_in is grouped and of bnsh or bsnh... kv_out is ungrouped and bsnh
template <typename T>
__global__ void UngroupAndTranspose(const T* kv_in,
                                    const T* kv_out,
                                    const int kv_num_heads,
                                    const bool is_bnsh) {
  const int h = threadIdx.x;
  const int out_n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int seqlen = gridDim.x;
  const int q_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int q_kv_head_ratio = q_num_heads / kv_num_heads;
  const int out_batch_stride = seqlen * q_num_heads * H;
  const int out_row_stride = q_num_heads * H;
  const int out_head_stride = H;

  const int in_batch_stride = seqlen * kv_num_heads * H;
  const int in_row_stride = is_bsnh ? kv_num_heads * H : H;
  const int in_head_stride = is_bnsh ? H : seqlen * H;
  const int in_n = out_n / q_kv_head_ratio;

  const int out_offset = out_batch_stride * b + out_row_stride * s + out_head_stride * out_n + h;
  const int in_offset = in_batch_stride * b + in_row_stride * s + in_head_stride * in_n + h;
  kv_out[out_offset] = kv_in[in_offset];
}

// Ungroup and tranpose present kv for use in Memory Efficient kernel
// TODO(aciddelgado): if T == float32, then H is head_size/2
template <typename T>
Status LaunchUngroupAndTranspose(contrib::GroupQueryAttentionParameters& parameters,
                                 GroupQueryAttentionData<T>& data) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.present_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);

  const int H = head_size / 4;
  if (H * num_heads <= max_threads_per_block) {
    const dim3 grid(sequence_length, batch_size, 1);
    const dim3 block(H, num_heads, 1);
    UngroupAndTranspose<float2><<< grid, block, 0, stream >>>(reinterpret_cast<const float2*>(data.present_key),
                                                              reinterpret_cast<const float2*>(data.present_k),
                                                              kv_num_heads,
                                                              past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    UngroupAndTranspose<float2><<< grid, block, 0, stream >>>(reinterpret_cast<const float2*>(data.present_value),
                                                              reinterpret_cast<const float2*>(data.present_v),
                                                              kv_num_heads,
                                                              past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  } else {
    assert(false);
  }
  // TODO(aciddelgado): can i have a large pls?
  // } else {
  //   const dim3 grid(present_sequence_length, batch_size, 1);
  //   const dim3 block(max_threads_per_block / kv_num_heads, kv_num_heads, 1);
  //   UngroupAndTransposeLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
  //                                                                 H,
  //                                                                 reinterpret_cast<const float2*>(data.past_key),
  //                                                                 reinterpret_cast<const float2*>(data.key),
  //                                                                 reinterpret_cast<float2*>(data.present_key),
  //                                                                 past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  //   UngroupAndTransposeLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
  //                                                                 H,
  //                                                                 reinterpret_cast<const float2*>(data.past_value),
  //                                                                 reinterpret_cast<const float2*>(data.value),
  //                                                                 reinterpret_cast<float2*>(data.present_value),
  //                                                                 past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
  // }
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

  assert(parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
  assert(parameters.num_heads % parameters.kv_num_heads == 0);

  void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
  void* key = reinterpret_cast<void*>(const_cast<T*>(data.key));
  void* value = reinterpret_cast<void*>(const_cast<T*>(data.value));

  bool is_causal = parameters.is_unidirectional;

  if (data.past_key == nullptr && data.present_key == nullptr) {
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
        device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
        parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, head_size,
        parameters.sequence_length, parameters.kv_sequence_length, scale, is_causal, parameters.num_splits,
        reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum)));
  // TODO(aciddelgado): refactor the next two if blocks into functions
  } else if (data.past_key == data.present_key) {
    // Assume past and present kv share buffer.
    assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
    assert(parameters.past_sequence_length >= 0);
    assert(data.past_value != nullptr);

    void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
    void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

    // Launch kernel to copy seqlen
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(batch_size) / thr_per_blk );
    repeat_seqlen<<< blk_in_grid, thr_per_blk, 0, stream >>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

    DUMP_TENSOR_INIT();
    DUMP_TENSOR("seqlens_k", data.seqlens_k, 1, batch_size);

    bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, stream, query, present_key, present_value, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
        reinterpret_cast<void*>(data.seqlens_k), batch_size, num_heads, kv_num_heads,
        head_size, sequence_length, present_sequence_length, kv_sequence_length,
        scale, is_causal, past_bsnh, parameters.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum),
        reinterpret_cast<void*>(data.out_accum)));

  } else if (data.present_key != nullptr && (data.past_key != nullptr || kv_sequence_length == present_sequence_length)) {
    assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
    // Note that Flash Attention kv-caching operates in place on a buffer... therefore this path is inneficient
    const int H = head_size / 4;
    if (H * kv_num_heads <= max_threads_per_block) {
      const dim3 grid(present_sequence_length, batch_size, 1);
      const dim3 block(H, kv_num_heads, 1);
      ConcatNewToPastKV<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                              reinterpret_cast<const float2*>(data.past_key),
                                                              reinterpret_cast<const float2*>(data.key),
                                                              reinterpret_cast<float2*>(data.present_key),
                                                              past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
      ConcatNewToPastKV<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                              reinterpret_cast<const float2*>(data.past_value),
                                                              reinterpret_cast<const float2*>(data.value),
                                                              reinterpret_cast<float2*>(data.present_value),
                                                              past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    } else {
      const dim3 grid(present_sequence_length, batch_size, 1);
      const dim3 block(max_threads_per_block / kv_num_heads, kv_num_heads, 1);
      ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(data.past_key),
                                                                    reinterpret_cast<const float2*>(data.key),
                                                                    reinterpret_cast<float2*>(data.present_key),
                                                                    past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
      ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(data.past_value),
                                                                    reinterpret_cast<const float2*>(data.value),
                                                                    reinterpret_cast<float2*>(data.present_value),
                                                                    past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
    }

    void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
    void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

    // Launch kernel to copy seqlen
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(batch_size) / thr_per_blk );
    repeat_seqlen<<< blk_in_grid, thr_per_blk, 0, stream >>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

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
  const int present_sequence_length = parameters.present_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat qkv_format = parameters.qkv_format;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  const void* query = data.query;
  const void* key = data.key;
  const void* value = data.value;
  if (data.present_key != nullptr) {
    if (data.past_key == data.present_key) {
      ConcatNewKVInPlace(data.present_key, data.key);
      ConcatNewKVInPlace(data.present_value, data.value);
    } else {
      LaunchConcatNewToPastKV(parameters, data);
    }
    // Can use present key and value directly if these conditions (required to run efficient attention) are met.
    if (num_heads == kv_num_heads && past_kv_format == AttentionQkvFormat::Q_K_V_BSNH) {
      key = data.present_key;
      value = data.present_value;
    } else {
      // Otherwise we use intermediate buffers to run memory efficient attention... BEST AVOID THIS PATH
      LaunchUngroupAndTranspose(parameters, data);
      key = data.present_k;
      value = data.present_v;
    }
  }

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_half = sizeof(T) == 2;
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.sequence_length = sequence_length;
  p.kv_sequence_length = data.present_key == nullptr ? kv_sequence_length : present_sequence_length;
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
  p.output = data.output;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(p.v_head_size, sizeof(T) == sizeof(float))
                    ? data.scratch
                    : nullptr;
  p.stream = stream;
  run_memory_efficient_attention(p);
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
  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(head_size)) : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    return FlashAttention(device_prop, stream, parameters, data);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    return EfficientAttention(device_prop, stream, parameters, data, scale);
  }
#endif

// #if USE_FLASH_ATTENTION
//   auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
//   const int max_threads_per_block = device_prop.maxThreadsPerBlock;
//   const int batch_size = parameters.batch_size;
//   const int sequence_length = parameters.sequence_length;
//   const int kv_sequence_length = parameters.kv_sequence_length;
//   const int present_sequence_length = parameters.present_sequence_length;
//   const int num_heads = parameters.num_heads;
//   const int kv_num_heads = parameters.kv_num_heads;
//   const int head_size = parameters.head_size;
//   AttentionQkvFormat qkv_format = parameters.qkv_format;
//   AttentionQkvFormat past_kv_format = parameters.past_kv_format;

//   // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
//   const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(head_size)) : parameters.scale;
//   if (data.use_flash_attention) {
//     assert(qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
//     assert(parameters.num_heads % parameters.kv_num_heads == 0);

//     void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
//     void* key = reinterpret_cast<void*>(const_cast<T*>(data.key));
//     void* value = reinterpret_cast<void*>(const_cast<T*>(data.value));

//     bool is_causal = parameters.is_unidirectional;

//     if (data.past_key == nullptr && data.present_key == nullptr) {
//       ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
//           device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
//           parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, head_size,
//           parameters.sequence_length, parameters.kv_sequence_length, scale, is_causal, parameters.num_splits,
//           reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum)));

//     } else if (data.past_key == data.present_key) {
//       // Assume past and present kv share buffer.
//       assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
//       assert(parameters.past_sequence_length >= 0);
//       assert(data.past_value != nullptr);

//       void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
//       void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

//       // Launch kernel to copy seqlen
//       int thr_per_blk = 256;
//       int blk_in_grid = ceil( float(batch_size) / thr_per_blk );
//       repeat_seqlen<<< blk_in_grid, thr_per_blk, 0, stream >>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

//       DUMP_TENSOR_INIT();
//       DUMP_TENSOR("seqlens_k", data.seqlens_k, 1, batch_size);

//       bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
//       ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
//           device_prop, stream, query, present_key, present_value, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
//           reinterpret_cast<void*>(data.seqlens_k), batch_size, num_heads, kv_num_heads,
//           head_size, sequence_length, present_sequence_length, kv_sequence_length,
//           scale, is_causal, past_bsnh, parameters.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum),
//           reinterpret_cast<void*>(data.out_accum)));

//     } else if (data.present_key != nullptr && (data.past_key != nullptr || kv_sequence_length == present_sequence_length)) {
//       assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
//       // Note that Flash Attention kv-caching operates in place on a buffer... therefore this path is inneficient
//       // TODO(aciddelgado): this only works for head_size % 4 == 0
//       const int H = head_size / 4;
//       if (H * kv_num_heads <= max_threads_per_block) {
//         const dim3 grid(present_sequence_length, batch_size, 1);
//         const dim3 block(H, kv_num_heads, 1);
//         ConcatNewToPastKV<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
//                                                                 reinterpret_cast<const float2*>(data.past_key),
//                                                                 reinterpret_cast<const float2*>(data.key),
//                                                                 reinterpret_cast<float2*>(data.present_key),
//                                                                 past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
//         ConcatNewToPastKV<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
//                                                                 reinterpret_cast<const float2*>(data.past_value),
//                                                                 reinterpret_cast<const float2*>(data.value),
//                                                                 reinterpret_cast<float2*>(data.present_value),
//                                                                 past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
//       } else {
//         const dim3 grid(present_sequence_length, batch_size, 1);
//         const dim3 block(max_threads_per_block / kv_num_heads, kv_num_heads, 1);
//         ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
//                                                                      H,
//                                                                      reinterpret_cast<const float2*>(data.past_key),
//                                                                      reinterpret_cast<const float2*>(data.key),
//                                                                      reinterpret_cast<float2*>(data.present_key),
//                                                                      past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
//         ConcatNewToPastKVLarge<float2><<< grid, block, 0, stream >>>(kv_sequence_length,
//                                                                      H,
//                                                                      reinterpret_cast<const float2*>(data.past_value),
//                                                                      reinterpret_cast<const float2*>(data.value),
//                                                                      reinterpret_cast<float2*>(data.present_value),
//                                                                      past_kv_format == AttentionQkvFormat::Q_K_V_BSNH);
//       }

//       void* present_key = reinterpret_cast<void*>(const_cast<T*>(data.present_key));
//       void* present_value = reinterpret_cast<void*>(const_cast<T*>(data.present_value));

//       // Launch kernel to copy seqlen
//       int thr_per_blk = 256;
//       int blk_in_grid = ceil( float(batch_size) / thr_per_blk );
//       repeat_seqlen<<< blk_in_grid, thr_per_blk, 0, stream >>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

//       bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
//       ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
//             device_prop, stream, query, present_key, present_value, data.output, reinterpret_cast<void*>(data.softmax_lse),
//             batch_size, num_heads, kv_num_heads, head_size,
//             sequence_length, present_sequence_length, scale, is_causal, parameters.num_splits,
//             reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum), past_bsnh));
//     }

//     DUMP_TENSOR_INIT();
//     DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, num_heads, head_size);

//     return Status::OK();
//   }
// #endif
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unfused Group Query Attention not implemented yet.");
}

// Template Instantiation
// template struct AttentionData<float>;

template struct GroupQueryAttentionData<half>;

// template Status QkvToContext<float>(
//     const cudaDeviceProp& device_prop,
//     cublasHandle_t& cublas,
//     Stream* ort_stream,
//     contrib::AttentionParameters& parameters,
//     AttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
