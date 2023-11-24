// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/decoder_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"

using namespace onnxruntime::contrib::attention_softmax_cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status DecoderQkvToContext(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
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
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int BN = batch_size * num_heads;
  const int BHN = BN * head_size;
  const int BNS = BN * sequence_length;
  const int k_buffer_offset = sequence_length * BHN;
  const int v_buffer_offset = (sequence_length + kv_sequence_length) * BHN;

  T* temp_qkv_buffer = workspace_buffer;
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());

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
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_key_cache, key_cache, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_value_cache, value_cache, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_key_cache, k, kv_sequence_length * BHN * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(new_value_cache, v, kv_sequence_length * BHN * sizeof(T),
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
        &zero, scratch1, kv_sequence_length, temp_matrix_size, BN, device_prop));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        kv_sequence_length, sequence_length, head_size,
        &alpha, k, head_size, strideA,
        q, head_size, strideB,
        &zero, scratch1, kv_sequence_length, temp_matrix_size, BN, device_prop));
  }

  constexpr bool is_unidirectional = false;
  const T* add_before_softmax = nullptr;
  if (has_key_padding_mask) {
    constexpr int mask_dimension = 2;
    constexpr int max_sequence_length = 0;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithRawMask<T>(
        ort_stream, kv_sequence_length, sequence_length, batch_size,
        num_heads, nullptr, key_padding_mask, add_before_softmax,
        false /*broadcast rpb*/, scratch1, scratch2, is_unidirectional,
        1.0f, mask_dimension, max_sequence_length, false, nullptr,
        mask_filter_value));
  } else {
    ORT_RETURN_IF_ERROR(ComputeSoftmax<T>(
        stream, kv_sequence_length, sequence_length, batch_size, num_heads,
        add_before_softmax, false /*broadcast rpb*/, scratch1, scratch2,
        is_unidirectional));
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (use_past && static_kv) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_size, sequence_length, kv_sequence_length,
        &one, value_cache, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        &zero, scratch3, head_size, strideB, BN, device_prop));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_size, sequence_length, kv_sequence_length,
        &one, v, head_size, strideA,
        scratch2, kv_sequence_length, temp_matrix_size,
        &zero, scratch3, head_size, strideB, BN, device_prop));
  }

  // scratch3 is BxNxSxH, transpose to output SxBxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads,
                        max_threads_per_block, true, scratch3, output);
}

Status LaunchDecoderAttentionKernel(
    const cudaDeviceProp& device_prop,
    Stream* stream,
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
        device_prop,
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
        device_prop,
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
