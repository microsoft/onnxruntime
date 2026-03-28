// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/linear_attention.h"
#include "contrib_ops/cuda/bert/linear_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;  // CudaKernel, Stream, GetDeviceProp, ToCudaType

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      LinearAttention,                                                \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("S", {DataTypeImpl::GetTensorType<float>(), \
                                DataTypeImpl::GetTensorType<T>()}),   \
      LinearAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
LinearAttention<T>::LinearAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t q_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("q_num_heads", &q_num_heads).IsOK() && q_num_heads > 0);
  q_num_heads_ = static_cast<int>(q_num_heads);

  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  chunk_size_ = static_cast<int>(chunk_size);
}

template <typename T>
Status LinearAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query_tensor = context->Input<Tensor>(0);
  const Tensor* key_tensor = context->Input<Tensor>(1);         // optional
  const Tensor* value_tensor = context->Input<Tensor>(2);       // optional
  const Tensor* past_state_tensor = context->Input<Tensor>(3);  // optional
  const Tensor* decay_tensor = context->Input<Tensor>(4);       // optional
  const Tensor* beta_tensor = context->Input<Tensor>(5);        // optional

  ORT_RETURN_IF_NOT(query_tensor != nullptr, "query input is required");

  const auto& query_shape = query_tensor->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3, "query must be 3D");

  const int batch_size = static_cast<int>(query_shape[0]);
  const int seq_len = static_cast<int>(query_shape[1]);
  const int query_hidden = static_cast<int>(query_shape[2]);

  // Determine d_k, d_v — packed QKV not supported on CUDA path (builder always provides separate Q,K,V)
  bool packed_qkv = (key_tensor == nullptr && value_tensor == nullptr);
  ORT_RETURN_IF_NOT(!packed_qkv, "Packed QKV not supported in CUDA LinearAttention; use separate Q, K, V inputs");
  ORT_RETURN_IF_NOT(key_tensor != nullptr && value_tensor != nullptr, "key and value are required");

  const auto& key_shape = key_tensor->Shape();
  const auto& value_shape = value_tensor->Shape();

  int d_k = query_hidden / q_num_heads_;
  int d_v = static_cast<int>(value_shape[2]) / kv_num_heads_;

  float s = scale_;
  if (s == 0.0f) {
    s = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  bool needs_decay = (update_rule_ == "gated" || update_rule_ == "gated_delta");
  bool needs_beta = (update_rule_ == "delta" || update_rule_ == "gated_delta");
  bool needs_retrieval = (update_rule_ == "delta" || update_rule_ == "gated_delta");

  bool decay_per_key_dim = false;
  if (decay_tensor != nullptr) {
    int64_t decay_last = decay_tensor->Shape()[2];
    decay_per_key_dim = (decay_last == kv_num_heads_ * d_k);
  }

  bool beta_per_head = false;
  if (beta_tensor != nullptr) {
    int64_t beta_last = beta_tensor->Shape()[2];
    beta_per_head = (beta_last == kv_num_heads_);
  }

  // Allocate outputs
  int output_hidden = q_num_heads_ * d_v;
  TensorShape output_shape({batch_size, seq_len, output_hidden});
  Tensor* output_tensor = context->Output(0, output_shape);

  TensorShape state_shape({batch_size, kv_num_heads_, d_k, d_v});
  Tensor* present_state_tensor = context->Output(1, state_shape);

  // If past_state is nullptr, zero-init present_state on device
  if (past_state_tensor == nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        present_state_tensor->MutableData<float>(), 0,
        static_cast<size_t>(batch_size) * kv_num_heads_ * d_k * d_v * sizeof(float),
        Stream(context)));
  } else {
    // Copy past_state -> present_state (will be updated in-place by kernel)
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        present_state_tensor->MutableData<float>(),
        past_state_tensor->Data<float>(),
        static_cast<size_t>(batch_size) * kv_num_heads_ * d_k * d_v * sizeof(float),
        cudaMemcpyDeviceToDevice,
        Stream(context)));
  }

  typedef typename OrtToCudaType<T>::type CudaT;

  return LaunchLinearAttentionKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(query_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(key_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(value_tensor->Data<T>()),
      decay_tensor ? reinterpret_cast<const CudaT*>(decay_tensor->Data<T>()) : nullptr,
      beta_tensor ? reinterpret_cast<const CudaT*>(beta_tensor->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()),
      present_state_tensor->MutableData<float>(),
      batch_size,
      seq_len,
      q_num_heads_,
      kv_num_heads_,
      d_k,
      d_v,
      s,
      needs_decay,
      decay_per_key_dim,
      needs_beta,
      beta_per_head,
      needs_retrieval,
      GetDeviceProp().maxThreadsPerBlock);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
