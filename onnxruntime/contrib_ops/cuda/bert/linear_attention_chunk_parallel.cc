// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/linear_attention_chunk_parallel.h"
#include "contrib_ops/cuda/bert/linear_attention_chunk_parallel_impl.h"
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
      LinearAttentionChunkParallel,                                   \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),     \
      LinearAttentionChunkParallel<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
LinearAttentionChunkParallel<T>::LinearAttentionChunkParallel(const OpKernelInfo& info)
    : CudaKernel(info) {
  std::string update_rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = StringToUpdateRule(update_rule_str);
  chunk_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("chunk_size", 64));
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status LinearAttentionChunkParallel<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* initial_state = context->Input<Tensor>(3);  // optional
  const Tensor* decay = context->Input<Tensor>(4);           // optional
  const Tensor* beta = context->Input<Tensor>(5);            // optional

  ORT_RETURN_IF_NOT(query != nullptr, "query input is required");
  ORT_RETURN_IF_NOT(key != nullptr, "key input is required");
  ORT_RETURN_IF_NOT(value != nullptr, "value input is required");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();

  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 4, "query must be 4D (B,H,T,d_k)");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == 4, "key must be 4D (B,H,T,d_k)");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 4, "value must be 4D (B,H,T,d_v)");

  int batch_size = static_cast<int>(q_shape[0]);
  int num_heads = static_cast<int>(q_shape[1]);
  int seq_len = static_cast<int>(q_shape[2]);
  int d_k = static_cast<int>(q_shape[3]);
  int d_v = static_cast<int>(v_shape[3]);

  ORT_RETURN_IF_NOT(k_shape[2] == seq_len, "key sequence length must match query");
  ORT_RETURN_IF_NOT(v_shape[2] == seq_len, "value sequence length must match query");
  ORT_RETURN_IF_NOT(d_k <= 128, "d_k must be <= 128 (kernel register limit)");
  ORT_RETURN_IF_NOT(d_v > 0 && d_k > 0, "d_k and d_v must be positive");

  if (initial_state != nullptr) {
    const auto& s_shape = initial_state->Shape();
    ORT_RETURN_IF_NOT(s_shape.NumDimensions() == 4, "initial_state must be 4D");
    ORT_RETURN_IF_NOT(s_shape[0] == batch_size && s_shape[1] == num_heads &&
                          s_shape[2] == d_k && s_shape[3] == d_v,
                      "initial_state shape must be (B,H,d_k,d_v)");
  }

  // Validate optional inputs based on update rule
  bool needs_decay = (update_rule_ == LinearAttentionUpdateRule::kGated ||
                      update_rule_ == LinearAttentionUpdateRule::kGatedDelta);
  bool needs_beta = (update_rule_ == LinearAttentionUpdateRule::kDelta ||
                     update_rule_ == LinearAttentionUpdateRule::kGatedDelta);

  ORT_RETURN_IF_NOT(!needs_decay || decay != nullptr,
                    "decay input is required for gated/gated_delta update rules");
  ORT_RETURN_IF_NOT(!needs_beta || beta != nullptr,
                    "beta input is required for delta/gated_delta update rules");

  bool decay_broadcasted = false;
  if (decay != nullptr) {
    const auto& decay_shape = decay->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 4, "decay must be 4D");
    decay_broadcasted = (decay_shape[3] == d_k);
  }

  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / sqrtf(static_cast<float>(d_k));
  }

  // Allocate outputs
  TensorShape output_shape({batch_size, num_heads, seq_len, d_v});
  Tensor* output = context->Output(0, output_shape);

  TensorShape state_shape({batch_size, num_heads, d_k, d_v});
  Tensor* final_state = context->Output(1, state_shape);

  // Calculate workspace size
  typedef typename ToCudaType<T>::MappedType CudaT;
  size_t element_size = sizeof(CudaT);
  size_t workspace_size = GetLinearAttentionChunkParallelWorkspaceSize(
      batch_size, num_heads, seq_len, d_k, d_v, chunk_size_, element_size);

  auto workspace = GetScratchBuffer<void>(workspace_size, context->GetComputeStream());

  return LaunchLinearAttentionChunkParallelKernel<CudaT>(
      Stream(context),
      update_rule_,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(value->Data<T>()),
      initial_state != nullptr ? reinterpret_cast<const CudaT*>(initial_state->Data<T>()) : nullptr,
      decay != nullptr ? reinterpret_cast<const CudaT*>(decay->Data<T>()) : nullptr,
      beta != nullptr ? reinterpret_cast<const CudaT*>(beta->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<CudaT*>(final_state->MutableData<T>()),
      workspace.get(),
      workspace_size,
      scale,
      batch_size,
      num_heads,
      seq_len,
      d_k,
      d_v,
      chunk_size_,
      decay_broadcasted);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
