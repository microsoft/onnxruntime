// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decoder_attention.h"
#include "attention_impl.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DecoderAttention,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DecoderAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
DecoderAttention<T>::DecoderAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  static_kv_ = info.GetAttrOrDefault<int64_t>("static_kv", 0) == 1;
  use_past_ = info.GetAttrOrDefault<int64_t>("use_past", 0) == 1;
  has_layer_state_ = info.GetAttrOrDefault<int64_t>("has_layer_state", 0) == 1;
}

template <typename T>
Status DecoderAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query(context->Input<Tensor>(0));
  const Tensor* key(context->Input<Tensor>(1));
  const Tensor* weights(context->Input<Tensor>(2));
  const Tensor* bias(context->Input<Tensor>(3));
  const Tensor* key_padding_mask(context->Input<Tensor>(4));
  const Tensor* key_cache(context->Input<Tensor>(5));
  const Tensor* value_cache(context->Input<Tensor>(6));

  auto& device_prop = GetDeviceProp();
  //TODO: check inputs

  // query shape (batch_size, sequence_length, input_hidden_size)
  const auto& query_shape = query->Shape();
  int sequence_length = static_cast<int>(query_shape[0]);
  int batch_size = static_cast<int>(query_shape[1]);
  int hidden_size = static_cast<int>(query_shape[2]);

  const auto& key_shape = key->Shape();
  int key_sequence_length = static_cast<int>(key_shape[0]);
  int head_size = hidden_size / num_heads_;

  //k, v sequence after gemm
  int kv_sequence_length = 0;

  // Generate q, k, v w/o cache
  // query input: (S, B, h1)
  // key input: (S', B, h1)
  // weight: (h1, h2)
  // h = N*H
  cublasHandle_t cublas = CublasHandle();
  constexpr size_t element_size = sizeof(T);

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  int m = 0, n = 0, k = 0;

  IAllocatorUniquePtr<T> gemm_buffer_p(nullptr);
  IAllocatorUniquePtr<T> gemm_query_buffer_p(nullptr);
  IAllocatorUniquePtr<T> gemm_kv_buffer_p(nullptr);

  // bugbug: need refactor
  if (!has_layer_state_ || !use_past_) {
    if (!static_kv_) {
      gemm_buffer_p = GetScratchBuffer<T>(3 * batch_size * sequence_length * hidden_size * element_size);
      m = sequence_length * batch_size;
      n = 3 * hidden_size;
      k = hidden_size;
      // broadcast bias: (3*h2, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
          GetConstOnes<CudaT>(m), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_buffer_p.get()), n, device_prop));
      // col-based
      // matmul: (3*h2, h1)*(h1, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
          reinterpret_cast<const CudaT*>(query->template Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_buffer_p.get()), n, device_prop));
      // gemm_buffer: (S*B, 3*h2)
      kv_sequence_length = sequence_length;
    } else {
      gemm_query_buffer_p = GetScratchBuffer<T>(batch_size * sequence_length * hidden_size * element_size);
      m = sequence_length * batch_size;
      n = hidden_size;
      k = hidden_size;
      // broadcast bias for query: (h2, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
          GetConstOnes<CudaT>(m), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_query_buffer_p.get()), n, device_prop));
      // matmul: (h2, h1)*(h1, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
          reinterpret_cast<const CudaT*>(query->template Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_query_buffer_p.get()), n, device_prop));
      // gemm_query_buffer in col-base: (h2, S*B)
      // Calculate k, v
      gemm_kv_buffer_p = GetScratchBuffer<T>(batch_size * 2 * key_sequence_length * hidden_size * element_size);
      m = key_sequence_length * batch_size;
      n = 2 * hidden_size;

      // broadcast bias for key and value: (2*h2, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->template Data<T>() + hidden_size), n,
          GetConstOnes<CudaT>(m), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // matmul: (2*h2, h1)*(h1, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(weights->template Data<T>() + hidden_size * hidden_size), n,
          reinterpret_cast<const CudaT*>(key->template Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // gemm_kv_buffer in col-base: (2*h2, T_S*B)
      kv_sequence_length = key_sequence_length;
    }
  } else {
    ORT_ENFORCE(nullptr != key_cache && nullptr != value_cache);
    // (B, N, S, H)
    const auto& cache_shape = key_cache->Shape();
    // key and value cache have identical shape
    int cache_sequence_length = static_cast<int>(cache_shape[2]);
    if (!static_kv_) {
      gemm_buffer_p = GetScratchBuffer<T>(3 * batch_size * sequence_length * hidden_size * element_size);
      m = sequence_length * batch_size;
      n = 3 * hidden_size;
      k = hidden_size;
      // broadcast bias: (3*h2, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
          GetConstOnes<CudaT>(m), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_buffer_p.get()), n, device_prop));
      // col-based
      // matmul: (3*h2, h1)*(h1, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
          reinterpret_cast<const CudaT*>(query->template Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_buffer_p.get()), n, device_prop));
      // gemm_buffer: (S*B, 3*h2)
      kv_sequence_length = cache_sequence_length + sequence_length;
    } else {
      gemm_query_buffer_p = GetScratchBuffer<T>(batch_size * sequence_length * hidden_size * element_size);
      m = sequence_length * batch_size;
      n = hidden_size;
      k = hidden_size;
      // broadcast bias for query: (h2, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
          GetConstOnes<CudaT>(m), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_query_buffer_p.get()), n, device_prop));
      // matmul: (h2, h1)*(h1, S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
          reinterpret_cast<const CudaT*>(query->template Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_query_buffer_p.get()), n, device_prop));
      // gemm_query_buffer in col-base: (h2, S*B)
      kv_sequence_length = cache_sequence_length;
    }
  }

  auto qkv_buffer_p = GetScratchBuffer<void>(batch_size * (sequence_length + 2 * kv_sequence_length) * hidden_size * element_size);
  auto workspace_p = GetScratchBuffer<void>(2 * batch_size * sequence_length * num_heads_ * element_size * (2 * head_size + kv_sequence_length));

  Tensor* output(context->Output(0, query_shape));
  TensorShape new_cache_shape({batch_size, num_heads_, kv_sequence_length, head_size});
  Tensor* new_key_cache(context->Output(1, new_cache_shape));
  Tensor* new_value_cache(context->Output(2, new_cache_shape));

  if (!LaunchDecoderAttentionKernel(
          device_prop,
          Stream(),
          cublas,
          element_size,
          batch_size,
          sequence_length,
          kv_sequence_length,
          num_heads_,
          head_size,
          static_kv_,
          use_past_,
          has_layer_state_,
          nullptr == gemm_buffer_p? nullptr : reinterpret_cast<const CudaT*>(gemm_buffer_p.get()),
          nullptr == gemm_query_buffer_p? nullptr : reinterpret_cast<const CudaT*>(gemm_query_buffer_p.get()),
          nullptr == gemm_kv_buffer_p? nullptr : reinterpret_cast<const CudaT*>(gemm_kv_buffer_p.get()),
          nullptr == key_padding_mask ? nullptr : key_padding_mask->template Data<bool>(),
          nullptr == key_cache ? nullptr : key_cache->template Data<T>(),
          nullptr == value_cache ? nullptr : value_cache->template Data<T>(),
          qkv_buffer_p.get(),
          workspace_p.get(),
          output->template MutableData<T>(),
          nullptr == new_key_cache ? nullptr : new_key_cache->template MutableData<T>(),
          nullptr == new_value_cache ? nullptr : new_value_cache->template MutableData<T>())) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
