// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/linear_attention.h"
#include "contrib_ops/cuda/bert/linear_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"

#include <string>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
LinearAttentionUpdateRule ParseUpdateRuleCuda(const std::string& rule) {
  if (rule == "linear") {
    return LinearAttentionUpdateRule::kLinear;
  } else if (rule == "gated") {
    return LinearAttentionUpdateRule::kGated;
  } else if (rule == "delta") {
    return LinearAttentionUpdateRule::kDelta;
  } else if (rule == "gated_delta") {
    return LinearAttentionUpdateRule::kGatedDelta;
  }
  ORT_THROW("Unknown update_rule: ", rule, ". Must be one of: linear, gated, delta, gated_delta");
}

LinearAttentionUpdateRuleCuda ToImplRule(LinearAttentionUpdateRule rule) {
  switch (rule) {
    case LinearAttentionUpdateRule::kLinear:
      return LinearAttentionUpdateRuleCuda::kLinear;
    case LinearAttentionUpdateRule::kGated:
      return LinearAttentionUpdateRuleCuda::kGated;
    case LinearAttentionUpdateRule::kDelta:
      return LinearAttentionUpdateRuleCuda::kDelta;
    case LinearAttentionUpdateRule::kGatedDelta:
      return LinearAttentionUpdateRuleCuda::kGatedDelta;
  }
  return LinearAttentionUpdateRuleCuda::kGatedDelta;
}
}  // namespace

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

#undef REGISTER_KERNEL_TYPED

template <typename T>
LinearAttention<T>::LinearAttention(const OpKernelInfo& info) : CudaKernel(info) {
  std::string update_rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = ParseUpdateRuleCuda(update_rule_str);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status LinearAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_state = context->Input<Tensor>(3);  // optional
  const Tensor* decay = context->Input<Tensor>(4);        // optional
  const Tensor* beta = context->Input<Tensor>(5);         // optional

  ORT_RETURN_IF_NOT(query != nullptr, "query input is required");
  ORT_RETURN_IF_NOT(key != nullptr, "key input is required");
  ORT_RETURN_IF_NOT(value != nullptr, "value input is required");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();

  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 4, "query must be 4D: (B, H, T, d_k)");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == 4, "key must be 4D: (B, H, T, d_k)");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 4, "value must be 4D: (B, H, T, d_v)");

  const int batch_size = static_cast<int>(q_shape[0]);
  const int num_heads = static_cast<int>(q_shape[1]);
  const int seq_len = static_cast<int>(q_shape[2]);
  const int key_dim = static_cast<int>(q_shape[3]);
  const int value_dim = static_cast<int>(v_shape[3]);

  ORT_RETURN_IF_NOT(k_shape[0] == batch_size && k_shape[1] == num_heads &&
                        k_shape[2] == seq_len && k_shape[3] == key_dim,
                    "key shape must match query in (B, H, T, d_k)");
  ORT_RETURN_IF_NOT(v_shape[0] == batch_size && v_shape[1] == num_heads && v_shape[2] == seq_len,
                    "value shape must match query in (B, H, T)");

  if (past_state != nullptr) {
    const auto& ps_shape = past_state->Shape();
    ORT_RETURN_IF_NOT(ps_shape.NumDimensions() == 4, "past_state must be 4D: (B, H, d_k, d_v)");
    ORT_RETURN_IF_NOT(ps_shape[0] == batch_size && ps_shape[1] == num_heads &&
                          ps_shape[2] == key_dim && ps_shape[3] == value_dim,
                      "past_state shape must be (B, H, d_k, d_v)");
  }

  bool is_gated = (update_rule_ == LinearAttentionUpdateRule::kGated ||
                   update_rule_ == LinearAttentionUpdateRule::kGatedDelta);
  bool is_delta = (update_rule_ == LinearAttentionUpdateRule::kDelta ||
                   update_rule_ == LinearAttentionUpdateRule::kGatedDelta);

  int decay_key_dim = 0;
  if (is_gated) {
    ORT_RETURN_IF_NOT(decay != nullptr, "decay input is required for gated and gated_delta update rules");
    const auto& decay_shape = decay->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 4, "decay must be 4D");
    ORT_RETURN_IF_NOT(decay_shape[0] == batch_size && decay_shape[1] == num_heads &&
                          decay_shape[2] == seq_len,
                      "decay shape must match (B, H, T, ...)");
    ORT_RETURN_IF_NOT(decay_shape[3] == 1 || decay_shape[3] == key_dim,
                      "decay last dimension must be 1 or key_dim");
    decay_key_dim = static_cast<int>(decay_shape[3]);
  }

  if (is_delta) {
    ORT_RETURN_IF_NOT(beta != nullptr, "beta input is required for delta and gated_delta update rules");
    const auto& beta_shape = beta->Shape();
    ORT_RETURN_IF_NOT(beta_shape.NumDimensions() == 4, "beta must be 4D");
    ORT_RETURN_IF_NOT(beta_shape[0] == batch_size && beta_shape[1] == num_heads &&
                          beta_shape[2] == seq_len && beta_shape[3] == 1,
                      "beta shape must be (B, H, T, 1)");
  }

  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  }

  // Allocate outputs
  TensorShape output_shape({batch_size, num_heads, seq_len, value_dim});
  Tensor* output = context->Output(0, output_shape);

  TensorShape state_shape({batch_size, num_heads, key_dim, value_dim});
  Tensor* present_state = context->Output(1, state_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  LaunchLinearAttentionKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(value->Data<T>()),
      past_state != nullptr ? reinterpret_cast<const CudaT*>(past_state->Data<T>()) : nullptr,
      decay != nullptr ? reinterpret_cast<const CudaT*>(decay->Data<T>()) : nullptr,
      beta != nullptr ? reinterpret_cast<const CudaT*>(beta->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_state->MutableData<T>()),
      batch_size,
      num_heads,
      seq_len,
      key_dim,
      value_dim,
      decay_key_dim,
      scale,
      ToImplRule(update_rule_));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
