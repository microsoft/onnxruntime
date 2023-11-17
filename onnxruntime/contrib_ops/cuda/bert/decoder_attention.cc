// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/decoder_attention.h"
#include "contrib_ops/cuda/bert/decoder_attention_impl.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include "core/framework/op_kernel.h"
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

namespace {

Status CheckInputs(const TensorShape& query_shape,
                   const TensorShape& key_shape,
                   const TensorShape& q_weights_shape,
                   const TensorShape& kv_weights_shape,
                   const TensorShape& bias_shape,
                   const Tensor* key_padding_mask,
                   const Tensor* key_cache,
                   const Tensor* value_cache,
                   const bool static_kv,
                   const bool use_past,
                   const bool has_layer_state,
                   const bool has_key_padding_mask) {
  const auto& query_shape_dims = query_shape.GetDims();
  if (query_shape_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_shape_dims.size());
  }

  int sequence_length = static_cast<int>(query_shape_dims[0]);
  int batch_size = static_cast<int>(query_shape_dims[1]);
  int hidden_size = static_cast<int>(query_shape_dims[2]);

  const auto& key_shape_dims = key_shape.GetDims();
  if (key_shape_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                           key_shape_dims.size());
  }
  int kv_sequence_length = static_cast<int>(key_shape_dims[0]);

  if (query_shape_dims[1] != key_shape_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query and key shall have the same batch size");
  }

  if (query_shape_dims[2] != key_shape_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query and key shall have the same hidden size");
  }

  const auto& q_weights_dims = q_weights_shape.GetDims();
  if (q_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'q_weights' is expected to have 2 dimensions, got ",
                           q_weights_dims.size());
  }

  const auto& kv_weights_dims = kv_weights_shape.GetDims();
  if (kv_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'kv_weights' is expected to have 2 dimensions, got ",
                           kv_weights_dims.size());
  }

  if (q_weights_dims[0] != hidden_size || q_weights_dims[1] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "q_weights shall have shape (hidden size, hidden size)");
  }

  if (kv_weights_dims[0] != hidden_size || kv_weights_dims[1] != 2 * static_cast<int64_t>(hidden_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "kv_weights shall have shape (hidden size, 2 * hidden size)");
  }

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  if (bias_dims[0] != 3 * static_cast<int64_t>(hidden_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bias shall have shape (3 * hidden size)");
  }

  int key_length = kv_sequence_length;
  if (key_padding_mask != nullptr && has_key_padding_mask == true) {
    const auto& kp_mask_dims = key_padding_mask->Shape().GetDims();

    if (kp_mask_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key_padding_mask' is expected to have 2 dimension, got ",
                             kp_mask_dims.size());
    }

    if (kp_mask_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "key_padding_mask shall have same batch size with query");
    }

    if (!has_layer_state || !use_past) {
      if (!static_kv) {
        key_length = sequence_length;
      }
    } else {
      if (!static_kv) {
        key_length = sequence_length + kv_sequence_length;
      }
    }

    if (kp_mask_dims[1] != key_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "key_padding_mask shall have same sequence length as generated key");
    }
  }

  if (key_cache != nullptr && value_cache != nullptr && has_layer_state && use_past) {
    const auto& key_cache_dims = key_cache->Shape().GetDims();
    if (key_cache_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key_cache' is expected to have 4 dimension, got ",
                             key_cache_dims.size());
    }

    const auto& value_cache_dims = value_cache->Shape().GetDims();
    if (value_cache_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'value_cache' is expected to have 4 dimension, got ",
                             value_cache_dims.size());
    }

    if (key_cache_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "key_cache shall have same batch size as query");
    }

    if (value_cache_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "value_cache shall have same batch size as query");
    }

    if (key_cache_dims[1] * key_cache_dims[3] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "key_cache shall have correct hidden size");
    }

    if (value_cache_dims[1] * value_cache_dims[3] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "value_cache shall have correct hidden size");
    }
  }

  return Status::OK();
}
}  // anonymous namespace

template <typename T>
DecoderAttention<T>::DecoderAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
}

template <typename T>
Status DecoderAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query(context->Input<Tensor>(0));
  const Tensor* key(context->Input<Tensor>(1));
  const Tensor* q_weights(context->Input<Tensor>(2));
  const Tensor* kv_weights(context->Input<Tensor>(3));
  const Tensor* bias(context->Input<Tensor>(4));
  const Tensor* key_padding_mask(context->Input<Tensor>(5));
  const Tensor* key_cache(context->Input<Tensor>(6));
  const Tensor* value_cache(context->Input<Tensor>(7));
  const Tensor* static_kv(context->Input<Tensor>(8));
  const Tensor* use_past(context->Input<Tensor>(9));
  const Tensor* has_layer_state(context->Input<Tensor>(10));
  const Tensor* has_key_padding_mask(context->Input<Tensor>(11));

  cudaStream_t stream = Stream(context);

  // Copy static_kv, use_past and has_layer_state to CPU
  auto pinned_buffer = AllocateBufferOnCPUPinned<void>(4 * sizeof(bool));
  bool* kernel_state_pinned = reinterpret_cast<bool*>(pinned_buffer.get());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(kernel_state_pinned, static_kv->Data<bool>(), sizeof(bool),
                                       cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(kernel_state_pinned + 1, use_past->Data<bool>(), sizeof(bool),
                                       cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(kernel_state_pinned + 2, has_layer_state->Data<bool>(), sizeof(bool),
                                       cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(kernel_state_pinned + 3, has_key_padding_mask->Data<bool>(), sizeof(bool),
                                       cudaMemcpyDeviceToHost, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();

  CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, stream));

  auto& device_prop = GetDeviceProp();

  // query shape (batch_size, sequence_length, input_hidden_size)
  const auto& query_shape = query->Shape();
  int sequence_length = static_cast<int>(query_shape[0]);
  int batch_size = static_cast<int>(query_shape[1]);
  int hidden_size = static_cast<int>(query_shape[2]);

  const auto& key_shape = key->Shape();
  int key_sequence_length = static_cast<int>(key_shape[0]);
  int head_size = hidden_size / num_heads_;

  // k, v sequence after gemm
  int kv_sequence_length = 0;

  // Generate q, k, v w/o cache
  // query input: (S, B, h1)
  // key input: (S', B, h1)
  // weight: (h1, h2)
  // h = N*H
  cublasHandle_t cublas = GetCublasHandle(context);
  constexpr size_t element_size = sizeof(T);

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  int m = 0, n = 0, k = 0;
  IAllocatorUniquePtr<T> gemm_query_buffer_p(nullptr);
  IAllocatorUniquePtr<T> gemm_kv_buffer_p(nullptr);

  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));
  bool static_kv_ = *kernel_state_pinned;
  bool use_past_ = *(kernel_state_pinned + 1);
  bool has_layer_state_ = *(kernel_state_pinned + 2);
  bool has_key_padding_mask_ = *(kernel_state_pinned + 3);

  ORT_RETURN_IF_ERROR(
      CheckInputs(query->Shape(),
                  key->Shape(),
                  q_weights->Shape(),
                  kv_weights->Shape(),
                  bias->Shape(),
                  key_padding_mask,
                  key_cache,
                  value_cache,
                  static_kv_,
                  use_past_,
                  has_layer_state_,
                  has_key_padding_mask_));

  // calculate q
  gemm_query_buffer_p = GetScratchBuffer<T>(static_cast<size_t>(batch_size) * sequence_length * hidden_size,
                                            context->GetComputeStream());
  m = sequence_length * batch_size;
  n = hidden_size;
  k = hidden_size;

  // TODO(tianleiwu): fuse bias and transpose
  // broadcast bias for query: (h2, S*B)
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
      reinterpret_cast<const CudaT*>(bias->Data<T>()), n,
      GetConstOnes<CudaT>(m, Stream(context)), 1,
      &zero, reinterpret_cast<CudaT*>(gemm_query_buffer_p.get()), n, device_prop));
  // matmul: (h2, h1)*(h1, S*B)
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(q_weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(query->Data<T>()), k,
      &one, reinterpret_cast<CudaT*>(gemm_query_buffer_p.get()), n, device_prop));
  // gemm_query_buffer in col-base: (h2, S*B)

  // calcualte k, v
  n = 2 * hidden_size;
  k = hidden_size;
  if (!has_layer_state_ || !use_past_) {
    if (!static_kv_) {
      gemm_kv_buffer_p = GetScratchBuffer<T>(static_cast<size_t>(batch_size) * 2 * sequence_length * hidden_size,
                                             context->GetComputeStream());
      m = sequence_length * batch_size;
      n = 2 * hidden_size;
      k = hidden_size;
      kv_sequence_length = sequence_length;
      // broadcast bias for key and value: (2*h2, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->Data<T>() + hidden_size), n,
          GetConstOnes<CudaT>(m, Stream(context)), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // matmul: (2*h2, h1)*(h1, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(kv_weights->Data<T>()), n,
          reinterpret_cast<const CudaT*>(query->Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // gemm_kv_buffer in col-base: (2*h2, T_S*B)
    } else {
      gemm_kv_buffer_p = GetScratchBuffer<T>(static_cast<size_t>(batch_size) * 2 * key_sequence_length * hidden_size,
                                             context->GetComputeStream());
      m = key_sequence_length * batch_size;
      n = 2 * hidden_size;
      k = hidden_size;
      kv_sequence_length = key_sequence_length;
      // broadcast bias for key and value: (2*h2, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->Data<T>() + hidden_size), n,
          GetConstOnes<CudaT>(m, Stream(context)), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // matmul: (2*h2, h1)*(h1, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(kv_weights->Data<T>()), n,
          reinterpret_cast<const CudaT*>(key->Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // gemm_kv_buffer in col-base: (2*h2, T_S*B)
    }
  } else {
    ORT_ENFORCE(nullptr != key_cache && nullptr != value_cache);  // (B, N, S, H)
    const auto& cache_shape = key_cache->Shape();
    // key and value cache have identical shape
    int cache_sequence_length = static_cast<int>(cache_shape[2]);
    if (!static_kv_) {
      gemm_kv_buffer_p = GetScratchBuffer<T>(static_cast<size_t>(batch_size) * 2 * sequence_length * hidden_size,
                                             context->GetComputeStream());
      m = sequence_length * batch_size;
      kv_sequence_length = cache_sequence_length + sequence_length;
      // broadcast bias for key and value: (2*h2, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
          reinterpret_cast<const CudaT*>(bias->Data<T>() + hidden_size), n,
          GetConstOnes<CudaT>(m, Stream(context)), 1,
          &zero, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // matmul: (2*h2, h1)*(h1, T_S*B)
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
          reinterpret_cast<const CudaT*>(kv_weights->Data<T>()), n,
          reinterpret_cast<const CudaT*>(query->Data<T>()), k,
          &one, reinterpret_cast<CudaT*>(gemm_kv_buffer_p.get()), n, device_prop));
      // gemm_kv_buffer in col-base: (2*h2, T_S*B)
    } else {
      kv_sequence_length = cache_sequence_length;
    }
  }

  size_t bytes = element_size * batch_size *
                 (static_cast<size_t>(sequence_length) + static_cast<size_t>(2) * kv_sequence_length) * hidden_size;
  auto qkv_buffer_p = GetScratchBuffer<void>(bytes, context->GetComputeStream());

  bytes = element_size * 2 * batch_size * sequence_length * num_heads_ *
          (static_cast<size_t>(2) * head_size + static_cast<size_t>(kv_sequence_length));
  auto workspace_p = GetScratchBuffer<void>(bytes, context->GetComputeStream());

  Tensor* output(context->Output(0, query_shape));
  TensorShape new_cache_shape({batch_size, num_heads_, kv_sequence_length, head_size});
  Tensor* new_key_cache(context->Output(1, new_cache_shape));
  Tensor* new_value_cache(context->Output(2, new_cache_shape));

  return LaunchDecoderAttentionKernel(
      device_prop,
#ifdef USE_ROCM
      GetTuningContext(),
#endif
      context->GetComputeStream(),
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
      has_key_padding_mask_,
      mask_filter_value_,
      nullptr == gemm_query_buffer_p ? nullptr : reinterpret_cast<const CudaT*>(gemm_query_buffer_p.get()),
      nullptr == gemm_kv_buffer_p ? nullptr : reinterpret_cast<const CudaT*>(gemm_kv_buffer_p.get()),
      nullptr == key_padding_mask ? nullptr : key_padding_mask->Data<bool>(),
      nullptr == key_cache ? nullptr : key_cache->Data<T>(),
      nullptr == value_cache ? nullptr : value_cache->Data<T>(),
      qkv_buffer_p.get(),
      workspace_p.get(),
      output->MutableData<T>(),
      nullptr == new_key_cache ? nullptr : new_key_cache->MutableData<T>(),
      nullptr == new_value_cache ? nullptr : new_value_cache->MutableData<T>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
