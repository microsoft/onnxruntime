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

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cpu/bert/attention_base.h"

using namespace onnxruntime::cuda;
using namespace cub;

#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace cuda {

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t total_sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * total_sequence_length;

  constexpr size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    size_t kv_sequence_length,
    size_t total_sequence_length,
    void* fused_runner) {
  const size_t qkv_size = element_size * batch_size * num_heads *
                          ((sequence_length * kv_sequence_length) * qk_head_size + kv_sequence_length * v_head_size);

  if (fused_runner != nullptr) {
    // Offsets without padding is B + 1. When we add padding, the size need to increase to 2B + 1.
    size_t sequenceOffsetBytes = sizeof(int) * (batch_size + 1);
    return qkv_size + reinterpret_cast<MHARunner*>(fused_runner)->getWorkspaceSize() + sequenceOffsetBytes;
  }

  return qkv_size + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length,
                                                total_sequence_length);
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    void* fused_runner) {
  constexpr size_t element_size = sizeof(T);
  const int max_threads_per_block = prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int total_sequence_length = parameters.total_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  T* qkv = data.workspace;
  const int batches = batch_size * num_heads;
  const int size_per_batch_q = sequence_length * qk_head_size;
  const int size_per_batch_k = kv_sequence_length * qk_head_size;
  const int size_per_batch_v = kv_sequence_length * v_head_size;
  const size_t elements_q = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_q);
  const size_t elements_k = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_k);
  const size_t elements_v = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_v);

  // Q, K and V pointers when fused attention if not used
  T* q = qkv;
  T* k = q + elements_q;
  T* v = k + elements_k;

  // For fused TRT attention, qkv need transpose to BxSxNx3xH
  bool use_fused_kernel = (nullptr != fused_runner && data.bias != nullptr);

  if (nullptr != data.gemm_buffer) {
    if (data.bias == nullptr) {
      // gemm_buffer should be BxSx3xNxH => qkv: 3xBxNxSxH
      ORT_ENFORCE(qk_head_size == v_head_size);
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 3, sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.gemm_buffer, qkv));
    } else {
      const int format = (use_fused_kernel ? 2 : 1);
      // format 1: BxSx(NH + NH + NH_v) => BxNxSxH + BxNxSxH + BxNxSxH_v
      // format 2: BxSx(NH + NH + NH) => BxSxNx(H + H + H)
      LaunchAddBiasTranspose(stream, 3, format, max_threads_per_block,
                             batch_size, sequence_length, num_heads, qk_head_size,
                             data.gemm_buffer, data.bias, qkv,
                             true, v_head_size);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    }
  } else {  // gemm_buffer == nullptr
    ORT_ENFORCE(data.query != nullptr && data.key != nullptr && data.value != nullptr && data.bias != nullptr);

  if (use_fused_kernel) {
      ORT_ENFORCE(sequence_length == kv_sequence_length && qk_head_size == v_head_size);

      // Q(BxSxNxH), K (BxSxNxH), V(BxSxNxH) => BxSxNx(H + H + H)
      LaunchAddBiasTransposeTrt(
        stream, max_threads_per_block,
        batch_size, sequence_length,
        num_heads, qk_head_size,
        data.bias, data.query, data.key, data.value, qkv);
    } else {
      // Query(BxSxNxH) => Q (BxNxSxH)
      constexpr int format = 0;
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, sequence_length, num_heads, qk_head_size,
                                data.query, data.bias, q,
                                true, -1);

      // Key (BxLxNxH) => K (BxNxLxH)
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, kv_sequence_length, num_heads, qk_head_size,
                                data.key, data.bias + num_heads * qk_head_size, k,
                                true, -1);

      // Value (BxLxNxH_v) => K (BxNxLxH_v)
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, kv_sequence_length, num_heads, v_head_size,
                                data.value, data.bias + 2 * num_heads * qk_head_size, v,
                                true, -1);
    }

    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  T* scratch1;
  scratch1 = qkv + elements_q + elements_k + elements_v;

  T* temp_output = scratch1;
  if (use_fused_kernel) {
    int* sequence_offset = reinterpret_cast<int*>(qkv + elements_q + elements_k + elements_v);
    LaunchTrtSequenceOffset(sequence_offset, data.mask_index, batch_size, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(fused_runner);
    fused_fp16_runner->setup(sequence_length, batch_size);

    fused_fp16_runner->run(qkv, nullptr, sequence_offset, data.output, nullptr, stream);

    return Status::OK();
  }

  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, total_sequence_length);
  T* scratch2 = scratch1 + (bytes / element_size);

  cublasSetStream(cublas, stream);

  // Concat past key value to present (2xBxNxLxH), where L is kv_sequence_length and T is total_sequence_length.
  // past_k (BxNxPxH) + k (BxNxLxH) => present_k (BxNxTxH)
  // past_v (BxNxPxH) + v (BxNxLxH) => present_v (BxNxTxH)
  // When there is past state, the head size for Q/K/V shall be same: H == H_v.
  const int present_size_per_batch_k = total_sequence_length * qk_head_size;
  const int present_size_per_batch_v = total_sequence_length * v_head_size;

  if (nullptr != data.present) {
    ORT_RETURN_IF_ERROR(
        LaunchConcatPastToPresent(stream, total_sequence_length, sequence_length, batch_size, qk_head_size, num_heads,
                                  max_threads_per_block, data.past, k, data.present));

    // Update pointers to present_k and present_v.
    k = data.present;
    v = data.present + batches * present_size_per_batch_k;
  }

  const int* mask_index = data.mask_index;
  gsl::span<const int64_t>& mask_index_dims = data.mask_index_dims;

  // Raw attention mask could be 2D (BxT) or 3D (BxSxT) or 4D(Bx1xMxM), where M is the max sequence length.
  bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxT
  // Q: BxNxSxH, K (present_k): BxNxTxH, Q*K': BxNxSxT
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(qk_head_size));
  const int temp_matrix_size = sequence_length * total_sequence_length;
  float one = 1.0f;
  float zero = 0.f;

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  float alpha = use_raw_attention_mask ? one : rsqrt_head_size;

  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N,
      total_sequence_length, sequence_length, qk_head_size,
      &alpha, k, qk_head_size, present_size_per_batch_k,
      q, qk_head_size, sequence_length * qk_head_size,
      &zero, scratch1, total_sequence_length, temp_matrix_size, batches, prop));

  // Apply softmax and store result R to scratch2: BxNxSxT
  if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
    const int mask_dimension = static_cast<int>(mask_index_dims.size());

    // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax used in Torch.
    const TransformerOptions* options = TransformerOptions::GetInstance();
    bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

    T* persistent_softmax_workspace = scratch1;  // replace Q*K' in place with masked score for persistent softmax.
    ORT_RETURN_IF_ERROR(
        ComputeSoftmaxWithRawMask<T>(stream, total_sequence_length, sequence_length, batch_size, num_heads,
                                     mask_index, nullptr, data.extra_add_qk, scratch1, scratch2,
                                     parameters.is_unidirectional, rsqrt_head_size, mask_dimension,
                                     parameters.max_sequence_length,
                                     use_persistent_softmax, persistent_softmax_workspace));
  } else if (nullptr != mask_index) {  // 1d mask index
    ORT_ENFORCE(mask_index_dims.size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
        stream, total_sequence_length, sequence_length, batch_size, num_heads,
        mask_index, mask_start, data.extra_add_qk, scratch1, scratch2, parameters.is_unidirectional));
  } else {  // no mask
    ORT_RETURN_IF_ERROR(
        ComputeSoftmax<T>(stream, total_sequence_length, sequence_length, batch_size, num_heads, data.extra_add_qk,
                          scratch1, scratch2, parameters.is_unidirectional));
  }

  // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  temp_output = qkv;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      v_head_size, sequence_length, total_sequence_length,
      &one, v, v_head_size, present_size_per_batch_v,
      scratch2, total_sequence_length, temp_matrix_size,
      &zero, temp_output, v_head_size, size_per_batch_v, batches, prop));

  // Temp_output is BxNxSxH_v, transpose to output BxSxNxH_v
  return LaunchTransCtx(stream, sequence_length, batch_size, v_head_size, num_heads,
                        max_threads_per_block, false, temp_output, data.output);
}

template <typename T>
Status DecoderQkvToContext(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    cublasHandle_t& cublas,
    const size_t element_size,
    const int batch_size,
    const int sequence_length,
    const int kv_sequence_length,
    const int num_heads,
    const int head_size,
    const bool static_kv,
    const bool use_past,
    const bool has_layer_state,
    const bool has_key_padding_mask,
    const T* gemm_query_buffer,
    const T* gemm_kv_buffer,
    const bool* key_padding_mask,
    const T* key_cache,
    const T* value_cache,
    T* qkv_buffer,
    T* workspace_buffer,
    T* output,
    T* new_key_cache,
    T* new_value_cache) {
  const int max_threads_per_block = prop.maxThreadsPerBlock;
  const int BN = batch_size * num_heads;
  const int BHN = BN * head_size;
  const int BNS = BN * sequence_length;
  const int k_buffer_offset = sequence_length * BHN;
  const int v_buffer_offset = (sequence_length + kv_sequence_length) * BHN;

  T* temp_qkv_buffer = workspace_buffer;

  const T* q = qkv_buffer;
  // transpose q and copy them to qkv_buffer
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, head_size, num_heads,
                                     max_threads_per_block, true, gemm_query_buffer, qkv_buffer));

  const T* k = qkv_buffer + k_buffer_offset;
  const T* v = qkv_buffer + v_buffer_offset;
  if (!has_layer_state || !use_past) {
    if (!static_kv) {
      // transpose kv and copy them to qkv_buffer
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 2, sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, true, gemm_kv_buffer, qkv_buffer + k_buffer_offset));
    } else {
      // transpose kv and copy them to qkv_buffer
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 2, kv_sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, true, gemm_kv_buffer, qkv_buffer + k_buffer_offset));
    }
  } else {
    if (!static_kv) {
      // transpose kv and copy them to temp_buffer
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 2, sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, true, gemm_kv_buffer, temp_qkv_buffer));
      // concat cache-k with k and copy to qkv_buffer
      if (nullptr != key_cache) {
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length,
                                                       sequence_length, batch_size, head_size, num_heads,
                                                       max_threads_per_block, 1,
                                                       key_cache,
                                                       temp_qkv_buffer,
                                                       qkv_buffer + k_buffer_offset));
      }
      // concat cache-v with v and copy to qkv_buffer
      if (nullptr != value_cache) {
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length,
                                                       sequence_length, batch_size, head_size, num_heads,
                                                       max_threads_per_block, 1,
                                                       value_cache,
                                                       temp_qkv_buffer + k_buffer_offset,
                                                       qkv_buffer + v_buffer_offset));
      }
    }
  }

  if (has_layer_state) {
    if (use_past && static_kv) {
      CHECK_CUDA(cudaMemcpyAsync(new_key_cache, key_cache, kv_sequence_length * BHN * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(new_value_cache, value_cache, kv_sequence_length * BHN * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(new_key_cache, k, kv_sequence_length * BHN * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(new_value_cache, v, kv_sequence_length * BHN * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream));
    }
  }

  // scratch1: BxNxSxL buffer
  // scratch2: BxNxSxL buffer
  // scratch3: BxNxSxH  buffer
  T* scratch1 = temp_qkv_buffer + 3 * BHN * sequence_length;
  T* scratch2 = scratch1 + BNS * kv_sequence_length;
  T* scratch3 = scratch2 + BNS * kv_sequence_length;

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxL
  // Q: BxNxSxH, K (present_k): BxNxLxH, Q*K': BxNxSxL
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * kv_sequence_length;
  float one = 1.0f;
  float zero = 0.f;

  float alpha = rsqrt_head_size;
  const int strideA = kv_sequence_length * head_size;
  const int strideB = sequence_length * head_size;
  if (use_past && static_kv) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        kv_sequence_length, sequence_length, head_size,
        &alpha, key_cache, head_size, strideA,
        q, head_size, strideB,
        &zero, scratch1, kv_sequence_length, temp_matrix_size, BN, prop));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        kv_sequence_length, sequence_length, head_size,
        &alpha, k, head_size, strideA,
        q, head_size, strideB,
        &zero, scratch1, kv_sequence_length, temp_matrix_size, BN, prop));
  }

  constexpr bool is_unidirectional = false;
  const T* add_before_softmax = nullptr;
  if (has_key_padding_mask) {
    constexpr int mask_dimension = 2;
    constexpr int max_sequence_length = 0;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(stream, kv_sequence_length, sequence_length, batch_size, num_heads,
                                                     nullptr, key_padding_mask, add_before_softmax, scratch1, scratch2,
                                                     is_unidirectional, 1.0f, mask_dimension, max_sequence_length,
                                                     false, nullptr));
  } else {
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(stream, kv_sequence_length, sequence_length, batch_size, num_heads,
                                          add_before_softmax, scratch1, scratch2, is_unidirectional));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (use_past && static_kv) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_size, sequence_length, kv_sequence_length,
        &one, value_cache, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        &zero, scratch3, head_size, strideB, BN, prop));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_size, sequence_length, kv_sequence_length,
        &one, v, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        &zero, scratch3, head_size, strideB, BN, prop));
  }

  // scratch3 is BxNxSxH, transpose to output SxBxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads,
                        max_threads_per_block, true, scratch3, output);
}

Status LaunchDecoderAttentionKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    cublasHandle_t& cublas,
    const size_t element_size,
    const int batch_size,
    const int sequence_length,
    const int kv_sequence_length,
    const int num_heads,
    const int head_size,
    const bool static_kv,
    const bool use_past,
    const bool has_layer_state,
    const bool has_key_padding_mask,
    const void* gemm_query_buffer,
    const void* gemm_kv_buffer,
    const bool* key_padding_mask,
    const void* key_cache,
    const void* value_cache,
    void* qkv_buffer,
    void* workspace_buffer,
    void* output,
    void* new_key_cache,
    void* new_value_cache) {
  if (element_size == 2) {
    return DecoderQkvToContext(
        prop,
        stream,
        cublas,
        element_size,
        batch_size,
        sequence_length,
        kv_sequence_length,
        num_heads,
        head_size,
        static_kv,
        use_past,
        has_layer_state,
        has_key_padding_mask,
        reinterpret_cast<const half*>(gemm_query_buffer),
        reinterpret_cast<const half*>(gemm_kv_buffer),
        key_padding_mask,
        reinterpret_cast<const half*>(key_cache),
        reinterpret_cast<const half*>(value_cache),
        reinterpret_cast<half*>(qkv_buffer),
        reinterpret_cast<half*>(workspace_buffer),
        reinterpret_cast<half*>(output),
        reinterpret_cast<half*>(new_key_cache),
        reinterpret_cast<half*>(new_value_cache));
  } else {
    return DecoderQkvToContext(
        prop,
        stream,
        cublas,
        element_size,
        batch_size,
        sequence_length,
        kv_sequence_length,
        num_heads,
        head_size,
        static_kv,
        use_past,
        has_layer_state,
        has_key_padding_mask,
        reinterpret_cast<const float*>(gemm_query_buffer),
        reinterpret_cast<const float*>(gemm_kv_buffer),
        key_padding_mask,
        reinterpret_cast<const float*>(key_cache),
        reinterpret_cast<const float*>(value_cache),
        reinterpret_cast<float*>(qkv_buffer),
        reinterpret_cast<float*>(workspace_buffer),
        reinterpret_cast<float*>(output),
        reinterpret_cast<float*>(new_key_cache),
        reinterpret_cast<float*>(new_value_cache));
  }
}

// Template Instantiation
template struct AttentionData<float>;

template struct AttentionData<half>;

template Status QkvToContext<float>(
    const cudaDeviceProp& prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data,
    void* fused_runner);

template Status QkvToContext<half>(
    const cudaDeviceProp& prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data,
    void* fused_runner);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
