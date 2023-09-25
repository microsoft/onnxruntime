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
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::attention_softmax_cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Kernel for seqlens_k
__global__ void repeat_seqlen(int32_t* seqlens_k, int32_t seqlen, int batch_size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id < batch_size) seqlens_k[id] = seqlen;
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data) {

  assert(data.use_flash_attention);

#if USE_FLASH_ATTENTION
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int max_sequence_length = parameters.max_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat qkv_format = parameters.qkv_format;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(head_size)) : parameters.scale;
  if (data.use_flash_attention) {
    assert(qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
    assert(parameters.num_heads % parameters.kv_num_heads == 0);

    void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
    void* key = reinterpret_cast<void*>(const_cast<T*>(data.key));
    void* value = reinterpret_cast<void*>(const_cast<T*>(data.value));

    bool is_causal = parameters.is_unidirectional;

    if (data.past_key == nullptr) {
      ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
          device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
          parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, head_size,
          parameters.sequence_length, parameters.total_sequence_length, scale, is_causal, parameters.num_splits,
          reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum)));
    } else {
      // Assume past and present kv share buffer.
      assert(past_kv_format == AttentionQkvFormat::Q_K_V_BSNH || past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
      assert(parameters.past_sequence_length >= 0);
      assert(data.past_value != nullptr);

      void* past_key = reinterpret_cast<void*>(const_cast<T*>(data.past_key));
      void* past_value = reinterpret_cast<void*>(const_cast<T*>(data.past_value));

      // Launch kernel to copy seqlen
      int thr_per_blk = 256;
      int blk_in_grid = ceil( float(batch_size) / thr_per_blk );
      repeat_seqlen<<< blk_in_grid, thr_per_blk, 0, stream >>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

      bool past_bsnh = past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
      ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
          device_prop, stream, query, past_key, past_value, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
          reinterpret_cast<void*>(data.seqlens_k), batch_size, num_heads, kv_num_heads,
          head_size, sequence_length, max_sequence_length, kv_sequence_length,
          scale, is_causal, past_bsnh, parameters.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum),
          reinterpret_cast<void*>(data.out_accum)));
    }


    DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, num_heads, head_size);

    return Status::OK();
  }
#endif
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
