/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
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
// (1) support GPT-2 past state, unidirectional mask and 4D attention mask from Megatron
// (2) support 2D attention mask
// (3) allow persistent softmax from PyTorch for debugging purpose.
// (4) support different input hidden size and model hidden size for pruned model
// (5) support different hidden sizes of Q/K and V
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
// #include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
// #include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
// #include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
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
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int max_sequence_length = parameters.max_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  AttentionQkvFormat qkv_format = AttentionQkvFormat::Q_K_V_BSNH;

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(head_size)) : parameters.scale;
  assert(data.use_flash_attention);
#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    assert(qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
    assert(parameters.num_heads % parameters.kv_num_heads == 0);

    void* query = reinterpret_cast<void*>(const_cast<T*>(data.query));
    void* key = reinterpret_cast<void*>(const_cast<T*>(data.key));
    void* value = reinterpret_cast<void*>(const_cast<T*>(data.value));

    bool is_causal = parameters.is_unidirectional;

    if (data.past_key == nullptr) {
      // TODO(aciddelgado): add support for concatenating past and kv to present kv when seqlens_k is not given
      ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
          device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
          parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, head_size,
          parameters.sequence_length, parameters.total_sequence_length, scale, is_causal, parameters.num_splits,
          reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum)));
    } else {
      // Assume past and present kv share buffer.
      assert(parameters.past_sequence_length >= 0);
      assert(data.past_value != nullptr);

      void* past_key = reinterpret_cast<void*>(const_cast<T*>(data.past_key));
      void* past_value = reinterpret_cast<void*>(const_cast<T*>(data.past_value));

      // Launch kernel to copy seqlen
      int thr_per_blk = 256;
      int blk_in_grid = ceil( float(batch_size) / thr_per_blk );
      repeat_seqlen<<< blk_in_grid, thr_per_blk, 0, stream >>>(data.seqlens_k, parameters.past_sequence_length, batch_size);

      ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
          device_prop, stream, query, past_key, past_value, key, value, data.output, reinterpret_cast<void*>(data.softmax_lse),
          reinterpret_cast<void*>(data.seqlens_k), batch_size, num_heads, kv_num_heads,
          head_size, sequence_length, max_sequence_length, kv_sequence_length,
          scale, is_causal, parameters.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum),
          reinterpret_cast<void*>(data.out_accum)));
    }


    DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, num_heads, head_size);

    return Status::OK();
  }
#endif
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unfused Group Query Attention not implemented yet.");

  // // The following are unfused attention.
  // assert(qkv_format == AttentionQkvFormat::Q_K_V_BNSH);

  // // Raw attention mask could be 2D (BxT) or 3D (BxSxT) or 4D(Bx1xMxM), where M is the max sequence length.
  // bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxT
  // // Q: BxNxSxH, K (present_k): BxNxTxH, Q*K': BxNxSxT
  // float one = 1.0f;
  // float zero = 0.f;

  // float alpha = use_raw_attention_mask ? one : scale;

  // cublasSetStream(cublas, stream);

  // DUMP_TENSOR_D("q[BNSH]", q, batch_size, num_heads, sequence_length, qk_head_size);
  // DUMP_TENSOR_D("k[BNSH]", k, batch_size, num_heads, total_sequence_length, qk_head_size);
  // CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
  //     cublas, CUBLAS_OP_T, CUBLAS_OP_N,
  //     total_sequence_length, sequence_length, qk_head_size,
  //     &alpha, k, qk_head_size, present_size_per_batch_k,
  //     q, qk_head_size, sequence_length * qk_head_size,
  //     &zero, scratch1, total_sequence_length, sequence_length * total_sequence_length, batches, device_prop));

  // DUMP_TENSOR_D("Q", q, batch_size, num_heads, sequence_length, qk_head_size);
  // DUMP_TENSOR_D("K", k, batch_size, num_heads, qk_head_size, sequence_length);
  // DUMP_TENSOR_D("QK", scratch1, batch_size, num_heads, sequence_length, total_sequence_length);

  // const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
  //                                              sequence_length, total_sequence_length);
  // T* scratch2 = scratch1 + (bytes / element_size);

  // // Apply softmax and store result R to scratch2: BxNxSxT
  // if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
  //   const int mask_dimension = static_cast<int>(mask_index_dims.size());

  //   // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax used in Torch.
  //   const TransformerOptions* options = TransformerOptions::GetInstance();
  //   bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

  //   T* persistent_softmax_workspace = scratch1;  // replace Q*K' in place with masked score for persistent softmax.
  //   ORT_RETURN_IF_ERROR(
  //       ComputeSoftmaxWithRawMask<T>(
  //           ort_stream, total_sequence_length, sequence_length, batch_size, num_heads,
  //           mask_index, nullptr, data.relative_position_bias, parameters.broadcast_res_pos_bias,
  //           scratch1, scratch2, parameters.is_unidirectional, scale, mask_dimension,
  //           parameters.max_sequence_length, use_persistent_softmax, persistent_softmax_workspace,
  //           mask_filter_value));
  // } else if (nullptr != mask_index) {  // 1d mask index
  //   assert(mask_index_dims.size() == 1);
  //   // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
  //   const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
  //   ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
  //       stream, total_sequence_length, sequence_length, batch_size, num_heads,
  //       mask_index, mask_start, data.relative_position_bias, parameters.broadcast_res_pos_bias,
  //       scratch1, scratch2, parameters.is_unidirectional));
  // } else {  // no mask
  //   ORT_RETURN_IF_ERROR(
  //       ComputeSoftmax<T>(
  //           stream, total_sequence_length, sequence_length, batch_size, num_heads, data.relative_position_bias,
  //           parameters.broadcast_res_pos_bias, scratch1, scratch2, parameters.is_unidirectional));
  // }

  // DUMP_TENSOR_D("Softmax", scratch2, batch_size, num_heads, sequence_length, total_sequence_length);
  // DUMP_TENSOR_D("V", v, batch_size, num_heads, sequence_length, v_head_size);

  // // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  // T* temp_output = qkv;
  // CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
  //     cublas, CUBLAS_OP_N, CUBLAS_OP_N,
  //     v_head_size, sequence_length, total_sequence_length,
  //     &one, v, v_head_size, present_size_per_batch_v,
  //     scratch2, total_sequence_length, sequence_length * total_sequence_length,
  //     &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop));

  // // Temp_output is BxNxSxH_v, transpose to output BxSxNxH_v
  // Status result = LaunchTransCtx(stream, sequence_length, batch_size, v_head_size, num_heads,
  //                                max_threads_per_block, false, temp_output, data.output);
  // DUMP_TENSOR("unfused output", data.output, batch_size, sequence_length, num_heads, v_head_size);
  // return result;
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
