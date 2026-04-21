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

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LinearAttention,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LinearAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LinearAttention<T>::LinearAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t q_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("q_num_heads", &q_num_heads).IsOK() && q_num_heads > 0);
  q_num_heads_ = static_cast<int>(q_num_heads);

  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  ORT_ENFORCE(update_rule_ == "linear" || update_rule_ == "gated" ||
                  update_rule_ == "delta" || update_rule_ == "gated_delta",
              "update_rule must be one of: linear, gated, delta, gated_delta");
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  // chunk_size_ reserved for future chunk-parallel prefill algorithm; not yet used.
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

  ORT_RETURN_IF_NOT(key_tensor != nullptr && value_tensor != nullptr, "key and value inputs are required");

  const auto& key_shape = key_tensor->Shape();
  const auto& value_shape = value_tensor->Shape();

  int d_k = query_hidden / q_num_heads_;
  int d_v = static_cast<int>(value_shape[2]) / kv_num_heads_;
  ORT_ENFORCE(static_cast<int>(key_shape[2]) % d_k == 0,
              "key last dim (", key_shape[2], ") must be divisible by d_k (", d_k, ")");
  int n_k_heads = static_cast<int>(key_shape[2]) / d_k;

  // GQA head mapping validations
  if (q_num_heads_ >= kv_num_heads_) {
    ORT_ENFORCE(q_num_heads_ % kv_num_heads_ == 0,
                "q_num_heads must be divisible by kv_num_heads");
  } else {
    ORT_ENFORCE(kv_num_heads_ % q_num_heads_ == 0,
                "kv_num_heads must be divisible by q_num_heads (inverse GQA)");
  }
  ORT_ENFORCE(kv_num_heads_ % n_k_heads == 0,
              "kv_num_heads must be divisible by n_k_heads");

  float s = scale_;
  if (s == 0.0f) {
    s = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  bool needs_decay = (update_rule_ == "gated" || update_rule_ == "gated_delta");
  bool needs_beta = (update_rule_ == "delta" || update_rule_ == "gated_delta");
  bool needs_retrieval = (update_rule_ == "delta" || update_rule_ == "gated_delta");

  ORT_ENFORCE(!needs_decay || decay_tensor != nullptr,
              "decay input is required for update_rule=", update_rule_);
  ORT_ENFORCE(!needs_beta || beta_tensor != nullptr,
              "beta input is required for update_rule=", update_rule_);

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
  int output_hidden = std::max(q_num_heads_, kv_num_heads_) * d_v;
  TensorShape output_shape({batch_size, seq_len, output_hidden});
  Tensor* output_tensor = context->Output(0, output_shape);

  TensorShape state_shape({batch_size, kv_num_heads_, d_k, d_v});
  Tensor* present_state_tensor = context->Output(1, state_shape);

  // If past_state is nullptr, zero-init present_state on device
  if (past_state_tensor == nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        present_state_tensor->MutableData<T>(), 0,
        static_cast<size_t>(batch_size) * kv_num_heads_ * d_k * d_v * sizeof(T),
        Stream(context)));
  } else {
    // Validate past_state shape matches expected (B, H_kv, d_k, d_v)
    const auto& past_shape = past_state_tensor->Shape();
    ORT_ENFORCE(past_shape.NumDimensions() == 4,
                "past_state must be rank 4 (B, H_kv, d_k, d_v), got rank ", past_shape.NumDimensions());
    ORT_ENFORCE(past_shape[0] == batch_size && past_shape[1] == kv_num_heads_ &&
                    past_shape[2] == d_k && past_shape[3] == d_v,
                "past_state shape mismatch: expected (", batch_size, ", ", kv_num_heads_, ", ", d_k, ", ", d_v,
                "), got (", past_shape[0], ", ", past_shape[1], ", ", past_shape[2], ", ", past_shape[3], ")");
    // Copy past_state -> present_state (will be updated in-place by kernel)
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        present_state_tensor->MutableData<T>(),
        past_state_tensor->Data<T>(),
        static_cast<size_t>(batch_size) * kv_num_heads_ * d_k * d_v * sizeof(T),
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
      reinterpret_cast<CudaT*>(present_state_tensor->MutableData<T>()),
      batch_size,
      seq_len,
      q_num_heads_,
      kv_num_heads_,
      n_k_heads,
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
