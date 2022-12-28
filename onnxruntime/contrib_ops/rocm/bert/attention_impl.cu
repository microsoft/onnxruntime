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
#include "core/platform/env_var_utils.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/attention_softmax.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"

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
    const float mask_filter_value,
    bool is_unidirectional,
    int past_sequence_length,
    const T* past,
    const T* extra_add_qk,
    T* present,
    bool use_persistent_softmax,
    bool use_gemm_rcr_bias_permute) {
  const int all_sequence_length = past_sequence_length + sequence_length;
  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, all_sequence_length);
  T* scratch1 = workspace;
  T* scratch2 = scratch1 + (bytes / element_size);
  T* scratch3 = scratch2 + (bytes / element_size);

  const int max_threads_per_block = prop.maxThreadsPerBlock;

  if (!use_gemm_rcr_bias_permute) {
  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size, num_heads,
                      max_threads_per_block, false, input, scratch3));
  }

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

  const bool use_batched_gemm_softmax_gemm_permute = ParseTestOnlyEnvironmentVariable<bool>("ORT_ATTENTION_USE_BATCHED_GEMM_SOFTMAX_GEMM_PERMUTE", {"0", "1"}) == true;

  if (use_batched_gemm_softmax_gemm_permute) {
    ORT_ENFORCE((std::is_same_v<T, MLFloat16>) || (std::is_same_v<T, half>) );
    ORT_ENFORCE(batch_size * num_heads == 768);
    ORT_ENFORCE(sequence_length == 512);
    ORT_ENFORCE(head_size == 64);

    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        2, 1, 1, 1, 1,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        // FIXME: it currently don't support bias add for GEMM1, so attention mask is not supported. Lets pretend that
        // ck supports it and we call it with no element masked out (all valid) for benchmarking purpose.
        ck::Tuple<>, // ck::Tuple<ck::half_t>, // acc0 bias datatype
        ck::Tuple<>,
        float,
        ck::half_t,  // CShuffleDType,

        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::ScaleAndResetNaNToMinusInfinity,

        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,

        ck::tensor_operation::device::GemmSpecialization::Default,

        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,

        1,

        256,  // block_size
        128,  // m_per_block
        256,  // n_per_block
        32,   // k_per_block
        64,   // Gemm1NPerBlock
        32,   // Gemm1KPerBlock
        8,    // ak1
        8,    // bk1
        2,    // b1k1
        32,   // m_per_xdl
        32,   // n_per_xdl
        1,    // m_xdl_per_wave
        8,    // n_xdl_per_wave
        2,    // Gemm1NXdlPerWave

        ck::Sequence<4, 64, 1>,  // thread_cluster_length
        ck::Sequence<1, 0, 2>,   // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,   // src_access_order
        2,                       // src_vector_dim
        8,                       // src_scalar_per_vector
        8,                       // dst_scalar_per_vector

        1,  // add_extra_dim

        ck::Sequence<4, 64, 1>,  // thread_cluster_length
        ck::Sequence<1, 0, 2>,   // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,   // src_access_order
        2,                       // src_vector_dim
        8,                       // src_scalar_per_vector
        8,                       // dst_scalar_per_vector

        1,  // add_extra_dim

        ck::Sequence<16, 16, 1>,  // thread_cluster_length
        ck::Sequence<0, 2, 1>,    // thread_cluster_arrange_order
        ck::Sequence<0, 2, 1>,    // src_access_order
        1,                        // src_vector_dim
        4,                        // src_scalar_per_vector
        2,                        // dst_scalar_per_vector

        0,  // add_extra_dim

        1,                                                                 // m_xdl_per_wave
        2,                                                                 // n_xdl_per_wave
        ck::Sequence<1, 32, 1, 8>,                                         // m_n_block_wave_per_xdl
        8,                                                                 // scalar_per_vector
        ck::tensor_operation::device::MaskingSpecialization::MaskDisabled  // causal_mask

        > batched_gemm_softmax_gemm_permute;

    LOGS_DEFAULT(ERROR) << "attn_mask is not used";

    int G0 = batch_size;
    int G1 = num_heads;
    int M = sequence_length;
    int N = sequence_length;
    int K = head_size;
    int O = head_size;

    auto invoker  = batched_gemm_softmax_gemm_permute.MakeInvoker();
    auto argument = batched_gemm_softmax_gemm_permute.MakeArgument(
        (const ck::half_t *)q,
        (const ck::half_t *)k,
        (const ck::half_t *)v,
        (ck::half_t *)output,
        {},
        {},
        {G0, G1, M, K},
        {G1 * M * K, M * K, K, 1},
        {G0, G1, N, K},
        {G1 * N * K, N * K, K, 1},
        {G0, G1, O, N},
        {G1 * N * O, N * O, 1, O},
        {G0, G1, M, O},
        {M * G1 * O, O, G1 * O, 1},
        {},
        {},
        {},
        {},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::ScaleAndResetNaNToMinusInfinity{rsqrt_head_size},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{});

    invoker.Run(argument, StreamConfig{stream, false});

    return Status::OK();
  } else {
  // clang-format off
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
                                      use_persistent_softmax, persistent_softmax_workspace, mask_filter_value));
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
  // clang-format on
  }
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
    const float mask_filter_value,
    const void* past,
    const void* extra_add_qk,
    void* workspace,
    void* output,
    void* present,
    bool use_gemm_rcr_bias_permute
    ) {
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
        mask_filter_value,
        is_unidirectional,
        past_sequence_length,
        reinterpret_cast<const __half*>(past),
        reinterpret_cast<const __half*>(extra_add_qk),
        reinterpret_cast<__half*>(present),
        use_persistent_softmax,
        use_gemm_rcr_bias_permute
        );
  } else {
    return QkvToContext(
        prop, tuning, rocblas, stream, batch_size, sequence_length, num_heads, head_size, element_size,
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        reinterpret_cast<float*>(workspace),
        mask_index,
        mask_index_dims,
        mask_filter_value,
        is_unidirectional,
        past_sequence_length,
        reinterpret_cast<const float*>(past),
        reinterpret_cast<const float*>(extra_add_qk),
        reinterpret_cast<float*>(present),
        use_persistent_softmax,
        use_gemm_rcr_bias_permute);
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
    const float mask_filter_value,
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
                                      false, 1, 2, static_cast<int>(0), false, nullptr, mask_filter_value));
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
    const float mask_filter_value,
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
        mask_filter_value,
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
        mask_filter_value,
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
