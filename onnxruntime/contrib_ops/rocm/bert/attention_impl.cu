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

// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/attention_softmax.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

using namespace onnxruntime::rocm;
using namespace hipcub;

namespace blas = onnxruntime::rocm::tunable::blas;

#define CHECK_ROCM(expr) HIP_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace rocm {

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t GetAttentionScratchSize(size_t element_size,
                               int batch_size,
                               int num_heads,
                               int sequence_length,
                               int all_sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * all_sequence_length;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int past_sequence_length) {
  size_t qkv_size = element_size * 3 * batch_size * sequence_length * num_heads * head_size;
  return qkv_size + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads,
                                                sequence_length, past_sequence_length + sequence_length);
}

template <typename T>
Status QkvToContext(
    const hipDeviceProp_t& prop,
    bool tuning,
    rocblas_handle& rocblas,
    hipStream_t stream,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const size_t element_size,
    const T* input,
    T* output,
    T* workspace,
    const int* mask_index,
    gsl::span<const int64_t> mask_index_dims,
    bool is_unidirectional,
    int past_sequence_length,
    const T* past,
    const T* extra_add_qk,
    T* present,
    bool use_persistent_softmax) {
  const int all_sequence_length = past_sequence_length + sequence_length;
  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, all_sequence_length);
  T* scratch1 = workspace;
  T* scratch2 = scratch1 + (bytes / element_size);
  T* scratch3 = scratch2 + (bytes / element_size);

  const int max_threads_per_block = prop.maxThreadsPerBlock;

  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size, num_heads,
                      max_threads_per_block, false, input, scratch3));

  // now scratch3 has Q, K, V: each has size BxNxSxH
  const int batches = batch_size * num_heads;
  const int size_per_batch = sequence_length * head_size;
  const int total_size = batches * size_per_batch;

  const T* q = scratch3;
  const T* k = q + total_size;
  const T* v = k + total_size;

  rocblas_set_stream(rocblas, stream);

  // Concat past (2xBxNxS'xH) to present (2xBxNxS*xH):
  // past_k (BxNxS'xH) + k (BxNxSxH) => present_k (BxNxS*xH)
  // past_v (BxNxS'xH) + v (BxNxSxH) => present_v (BxNxS*xH)
  const int present_size_per_batch = all_sequence_length * head_size;
  if (nullptr != present) {
    ORT_RETURN_IF_ERROR(
      LaunchConcatPastToPresent(stream, all_sequence_length, sequence_length, batch_size, head_size, num_heads,
                                   max_threads_per_block, past, k, present));

    // update pointers to present_k and present_v.
    k = present;
    v = present + batches * present_size_per_batch;
  }

  // Raw attention mask could be 2D (BxS) or 3D (BxSxS*) or 4D(Bx1xMxM), where M is the max sequence length.
  bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
  // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * all_sequence_length;

  ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
      tuning, stream, rocblas,
      blas::BlasOp::Trans, blas::BlasOp::NonTrans,
      all_sequence_length, sequence_length, head_size,
      // For raw attention mask, the scalar if 1/sqrt(H) is moved to softmax computation.
      /*alpha=*/use_raw_attention_mask ? 1.0f : rsqrt_head_size,
      k, head_size, present_size_per_batch,
      q, head_size, size_per_batch,
      /*beta=*/0.0f,
      scratch1, all_sequence_length, temp_matrix_size,
      batches));

  // apply softmax and store result P to scratch2: BxNxSxS*
  if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
    const int mask_dimension = static_cast<int>(mask_index_dims.size());
    const int max_sequence_length = mask_dimension == 4 ? static_cast<int>(mask_index_dims[3]) : 0;

    T* persistent_softmax_workspace = scratch1;  // replace Q*K' in place if persistent softmax is selected.
    ORT_RETURN_IF_ERROR(
        ComputeSoftmaxWithRawMask<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads,
                                      mask_index, nullptr, extra_add_qk, scratch1, scratch2,
                                      is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                      use_persistent_softmax, persistent_softmax_workspace));
  } else if (nullptr != mask_index) {  // 1d mask index
    ORT_ENFORCE(mask_index_dims.size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads,
                                     mask_index, mask_start, extra_add_qk, scratch1, scratch2, is_unidirectional));
  } else {  // no mask
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads,
                           extra_add_qk, scratch1, scratch2, is_unidirectional));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
      tuning, stream, rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      head_size, sequence_length, all_sequence_length,
      /*alpha=*/1.0f,
      v, head_size, present_size_per_batch,
      scratch2, all_sequence_length, temp_matrix_size,
      /*beta=*/0.0f,
      scratch3, head_size, size_per_batch,
      batches));

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads,
                        max_threads_per_block, false, scratch3, output);
}

Status LaunchAttentionKernel(
    const hipDeviceProp_t& prop,
    bool tuning,
    hipStream_t stream,
    rocblas_handle& rocblas,
    const size_t element_size,
    int batch_size,
    int sequence_length,
    int num_heads,
    int head_size,
    int past_sequence_length,
    bool is_unidirectional,
    const void* input,
    const int* mask_index,
    gsl::span<const int64_t> mask_index_dims,
    const void* past,
    const void* extra_add_qk,
    void* workspace,
    void* output,
    void* present) {
  // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax
  const TransformerOptions* options = TransformerOptions::GetInstance();
  bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();
  if (element_size == 2) {
    return QkvToContext(
        prop, tuning, rocblas, stream, batch_size, sequence_length, num_heads, head_size, element_size,
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<__half*>(output),
        reinterpret_cast<__half*>(workspace),
        mask_index,
        mask_index_dims,
        is_unidirectional,
        past_sequence_length,
        reinterpret_cast<const __half*>(past),
        reinterpret_cast<const __half*>(extra_add_qk),
        reinterpret_cast<__half*>(present),
        use_persistent_softmax);
  } else {
    return QkvToContext(
        prop, tuning, rocblas, stream, batch_size, sequence_length, num_heads, head_size, element_size,
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        reinterpret_cast<float*>(workspace),
        mask_index,
        mask_index_dims,
        is_unidirectional,
        past_sequence_length,
        reinterpret_cast<const float*>(past),
        reinterpret_cast<const float*>(extra_add_qk),
        reinterpret_cast<float*>(present),
        use_persistent_softmax);
  }
}

template <typename T>
Status DecoderQkvToContext(
    const hipDeviceProp_t& prop,
    bool tuning,
    hipStream_t stream,
    rocblas_handle& rocblas,
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
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, head_size,
                      num_heads, max_threads_per_block, true, gemm_query_buffer, qkv_buffer));

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
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length, sequence_length,
                                                              batch_size, head_size, num_heads,
                                                              max_threads_per_block, 1, key_cache,
                                                              temp_qkv_buffer, qkv_buffer + k_buffer_offset));
      }
      // concat cache-v with v and copy to qkv_buffer
      if (nullptr != value_cache) {
        ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(stream, kv_sequence_length, sequence_length,
                                                                batch_size, head_size, num_heads,
                                                                max_threads_per_block, 1, value_cache,
                                                                temp_qkv_buffer + k_buffer_offset,
                                                                qkv_buffer + v_buffer_offset));
      }
    }
  }

  if (has_layer_state) {
    if (use_past && static_kv) {
      CHECK_ROCM(hipMemcpyAsync(new_key_cache, key_cache,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
      CHECK_ROCM(hipMemcpyAsync(new_value_cache, value_cache,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
    } else {
      CHECK_ROCM(hipMemcpyAsync(new_key_cache, k,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
      CHECK_ROCM(hipMemcpyAsync(new_value_cache, v,
                                kv_sequence_length * BHN * sizeof(T), hipMemcpyDeviceToDevice, stream));
    }
  }

  // scratch1: BxNxSxS* buffer
  // scratch2: BxNxSxS* buffer
  // scratch3: BxNxSxH  buffer
  T* scratch1 = temp_qkv_buffer + 3 * BHN * sequence_length;
  T* scratch2 = scratch1 + BNS * kv_sequence_length;
  T* scratch3 = scratch2 + BNS * kv_sequence_length;

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
  // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * kv_sequence_length;

  const int strideA = kv_sequence_length * head_size;
  const int strideB = sequence_length * head_size;
  if (use_past && static_kv) {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning, stream, rocblas,
        blas::BlasOp::Trans, blas::BlasOp::NonTrans,
        kv_sequence_length, sequence_length, head_size,
        /*alpha=*/rsqrt_head_size,
        key_cache, head_size, strideA,
        q, head_size, strideB,
        /*beta=*/0.0f,
        scratch1, kv_sequence_length, temp_matrix_size,
        BN));
  } else {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning, stream, rocblas,
        blas::BlasOp::Trans, blas::BlasOp::NonTrans,
        kv_sequence_length, sequence_length, head_size,
        /*alpha=*/rsqrt_head_size,
        k, head_size, strideA,
        q, head_size, strideB,
        /*beta=*/0.0f,
        scratch1, kv_sequence_length, temp_matrix_size,
        BN));
  }

  if (has_key_padding_mask) {
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(stream, kv_sequence_length, sequence_length, batch_size,
                                      num_heads, nullptr, key_padding_mask, nullptr, scratch1, scratch2,
                                      false, 1, 2, static_cast<int>(0), false, nullptr));
  } else {
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(stream, kv_sequence_length, sequence_length, batch_size,
                           num_heads, nullptr, scratch1, scratch2, false));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (use_past && static_kv) {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning, stream, rocblas,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        head_size, sequence_length, kv_sequence_length,
        /*alpha=*/1.0f,
        value_cache, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        /*beta=*/0.0f,
        scratch3, head_size, strideB,
        BN));
  } else {
    ORT_RETURN_IF_ERROR(blas::column_major::StridedBatchedGemm(
        tuning, stream, rocblas,
        blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
        head_size, sequence_length, kv_sequence_length,
        /*alpha=*/1.0f,
        v, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        /*beta=*/0.0f,
        scratch3, head_size, strideB,
        BN));
  }

  // scratch3 is BxNxSxH, transpose to output SxBxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size,
                        num_heads, max_threads_per_block, true, scratch3, output);
}

Status LaunchDecoderAttentionKernel(
    const hipDeviceProp_t& prop,
    bool tuning,
    hipStream_t stream,
    rocblas_handle& rocblas,
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
        tuning,
        stream,
        rocblas,
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
        tuning,
        stream,
        rocblas,
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

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
