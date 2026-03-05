// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/linear_attention_recurrent.h"
#include "contrib_ops/cuda/bert/linear_attention_recurrent_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
LinearAttentionUpdateRule StringToUpdateRule(const std::string& s) {
  if (s == "linear") return LinearAttentionUpdateRule::kLinear;
  if (s == "gated") return LinearAttentionUpdateRule::kGated;
  if (s == "delta") return LinearAttentionUpdateRule::kDelta;
  if (s == "gated_delta") return LinearAttentionUpdateRule::kGatedDelta;
  ORT_THROW("Unknown linear attention update_rule: ", s);
}
}  // namespace

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      LinearAttentionRecurrent,                                       \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .MayInplace(3, 1),                                          \
      LinearAttentionRecurrent<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
LinearAttentionRecurrent<T>::LinearAttentionRecurrent(const OpKernelInfo& info)
    : CudaKernel(info) {
  std::string update_rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = StringToUpdateRule(update_rule_str);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status LinearAttentionRecurrent<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_state = context->Input<Tensor>(3);
  const Tensor* decay = context->Input<Tensor>(4);  // optional
  const Tensor* beta = context->Input<Tensor>(5);    // optional

  // Validate required inputs
  ORT_RETURN_IF_NOT(query != nullptr, "query input is required");
  ORT_RETURN_IF_NOT(key != nullptr, "key input is required");
  ORT_RETURN_IF_NOT(value != nullptr, "value input is required");
  ORT_RETURN_IF_NOT(past_state != nullptr, "past_state input is required");

  // Validate shapes: query/key (B,H,1,d_k), value (B,H,1,d_v), state (B,H,d_k,d_v)
  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();
  const auto& s_shape = past_state->Shape();

  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 4, "query must be 4D (B,H,1,d_k)");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == 4, "key must be 4D (B,H,1,d_k)");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 4, "value must be 4D (B,H,1,d_v)");
  ORT_RETURN_IF_NOT(s_shape.NumDimensions() == 4, "past_state must be 4D (B,H,d_k,d_v)");

  int batch_size = static_cast<int>(q_shape[0]);
  int num_heads = static_cast<int>(q_shape[1]);
  int d_k = static_cast<int>(q_shape[3]);
  int d_v = static_cast<int>(v_shape[3]);

  ORT_RETURN_IF_NOT(q_shape[2] == 1, "query sequence length must be 1 for recurrent mode");
  ORT_RETURN_IF_NOT(k_shape[2] == 1, "key sequence length must be 1 for recurrent mode");
  ORT_RETURN_IF_NOT(v_shape[2] == 1, "value sequence length must be 1 for recurrent mode");

  ORT_RETURN_IF_NOT(s_shape[0] == batch_size && s_shape[1] == num_heads &&
                        s_shape[2] == d_k && s_shape[3] == d_v,
                    "past_state shape must be (B,H,d_k,d_v)");

  // Validate optional inputs based on update rule
  bool needs_decay = (update_rule_ == LinearAttentionUpdateRule::kGated ||
                      update_rule_ == LinearAttentionUpdateRule::kGatedDelta);
  bool needs_beta = (update_rule_ == LinearAttentionUpdateRule::kDelta ||
                     update_rule_ == LinearAttentionUpdateRule::kGatedDelta);

  ORT_RETURN_IF_NOT(!needs_decay || decay != nullptr,
                    "decay input is required for gated/gated_delta update rules");
  ORT_RETURN_IF_NOT(!needs_beta || beta != nullptr,
                    "beta input is required for delta/gated_delta update rules");

  // Determine if decay is broadcasted (per-key-dim vs scalar)
  bool decay_broadcasted = false;
  if (decay != nullptr) {
    const auto& decay_shape = decay->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 4, "decay must be 4D");
    decay_broadcasted = (decay_shape[3] == d_k);
  }

  // Compute scale
  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / sqrtf(static_cast<float>(d_k));
  }

  // Allocate outputs
  TensorShape output_shape({batch_size, num_heads, 1, d_v});
  Tensor* output = context->Output(0, output_shape);

  TensorShape state_shape({batch_size, num_heads, d_k, d_v});
  Tensor* present_state = context->Output(1, state_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  return LaunchLinearAttentionRecurrentKernel<CudaT>(
      Stream(context),
      update_rule_,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(value->Data<T>()),
      reinterpret_cast<const CudaT*>(past_state->Data<T>()),
      decay != nullptr ? reinterpret_cast<const CudaT*>(decay->Data<T>()) : nullptr,
      beta != nullptr ? reinterpret_cast<const CudaT*>(beta->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_state->MutableData<T>()),
      scale,
      batch_size,
      num_heads,
      d_k,
      d_v,
      decay_broadcasted);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
