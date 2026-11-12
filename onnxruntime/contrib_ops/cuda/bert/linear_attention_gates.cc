// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/linear_attention_gates.h"
#include "contrib_ops/cuda/bert/linear_attention_gates_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::contrib::linear_attention_gates_helper;
using namespace onnxruntime::cuda;  // CudaKernel, OrtToCudaType

#define REGISTER_KERNEL_TYPED(Op, T)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      Op,                                                              \
      kMSDomain,                                                       \
      1,                                                               \
      T,                                                               \
      kCudaExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("TF", DataTypeImpl::GetTensorType<float>()), \
      Op<T>);

REGISTER_KERNEL_TYPED(LinearAttentionGate, float)
REGISTER_KERNEL_TYPED(LinearAttentionGate, MLFloat16)
REGISTER_KERNEL_TYPED(LinearAttentionGate, BFloat16)
REGISTER_KERNEL_TYPED(MoEGroupRouter, float)
REGISTER_KERNEL_TYPED(MoEGroupRouter, MLFloat16)
REGISTER_KERNEL_TYPED(MoEGroupRouter, BFloat16)

#undef REGISTER_KERNEL_TYPED

#define REGISTER_KERNEL_TYPED(Op, T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Op,                                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Op<T>);

REGISTER_KERNEL_TYPED(GatedRMSNorm, float)
REGISTER_KERNEL_TYPED(GatedRMSNorm, MLFloat16)
REGISTER_KERNEL_TYPED(GatedRMSNorm, BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
Status LinearAttentionGate<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename OrtToCudaType<T>::type CudaT;

  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* dt_bias = context->Input<Tensor>(1);
  const Tensor* decay_scale = context->Input<Tensor>(2);
  const Tensor* b = context->Input<Tensor>(3);  // optional

  const auto& a_shape = a->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "a must have rank >= 1");
  const int64_t num_heads = a_shape[a_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(num_heads > 0, "a last dimension must be positive");
  ORT_RETURN_IF_NOT(num_heads <= std::numeric_limits<int>::max(),
                    "a last dimension is too large for the CUDA kernel");

  ORT_RETURN_IF_NOT(dt_bias->Shape().Size() == num_heads,
                    "dt_bias must have ", num_heads, " elements, got ", dt_bias->Shape().Size());
  ORT_RETURN_IF_NOT(decay_scale->Shape().Size() == num_heads,
                    "decay_scale must have ", num_heads, " elements, got ", decay_scale->Shape().Size());

  Tensor* decay = context->Output(0, a_shape);
  Tensor* beta = context->Output(1, a_shape);

  if (beta != nullptr) {
    ORT_RETURN_IF_NOT(b != nullptr, "the b input is required when the beta output is requested");
    ORT_RETURN_IF_NOT(b->Shape() == a_shape, "b must have the same shape as a");
  }

  const int64_t num_tokens = a_shape.Size() / num_heads;
  return LaunchLinearAttentionGateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(decay->MutableData<T>()),
      beta == nullptr ? nullptr : reinterpret_cast<CudaT*>(beta->MutableData<T>()),
      reinterpret_cast<const CudaT*>(a->Data<T>()),
      b == nullptr ? nullptr : reinterpret_cast<const CudaT*>(b->Data<T>()),
      dt_bias->Data<float>(),
      decay_scale->Data<float>(),
      num_tokens,
      static_cast<int>(num_heads));
}

template <typename T>
GatedRMSNorm<T>::GatedRMSNorm(const OpKernelInfo& info) : CudaKernel(info) {
  activation_ = ParseGatedRMSNormActivationOrThrow(info.GetAttrOrDefault<std::string>("activation", "silu"));
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
}

template <typename T>
Status GatedRMSNorm<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename OrtToCudaType<T>::type CudaT;

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* scale = context->Input<Tensor>(1);
  const Tensor* gate = context->Input<Tensor>(2);

  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(gate->Shape() == shape, "gate must have the same shape as X");

  const int64_t norm_size = scale->Shape().Size();
  ORT_RETURN_IF_NOT(norm_size > 0, "scale must not be empty");
  ORT_RETURN_IF_NOT(norm_size <= std::numeric_limits<int>::max(),
                    "scale is too large for the CUDA kernel");
  const int64_t last_dim = shape[shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(last_dim % norm_size == 0,
                    "X last dimension (", last_dim, ") must be a multiple of the scale length (",
                    norm_size, ")");

  Tensor* output = context->Output(0, shape);
  const int64_t num_rows = shape.Size() / norm_size;

  return LaunchGatedRMSNormKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      reinterpret_cast<const CudaT*>(scale->Data<T>()),
      reinterpret_cast<const CudaT*>(gate->Data<T>()),
      num_rows,
      static_cast<int>(norm_size),
      epsilon_,
      activation_,
      GetDeviceProp().maxThreadsPerBlock);
}

template class LinearAttentionGate<float>;
template class LinearAttentionGate<MLFloat16>;
template class LinearAttentionGate<BFloat16>;
template class GatedRMSNorm<float>;
template class GatedRMSNorm<MLFloat16>;
template class GatedRMSNorm<BFloat16>;

template <typename T>
MoEGroupRouter<T>::MoEGroupRouter(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("top_k", &top_k_));
  n_group_ = info.GetAttrOrDefault<int64_t>("n_group", 1);
  topk_group_ = info.GetAttrOrDefault<int64_t>("topk_group", 1);
  group_topk_ = info.GetAttrOrDefault<int64_t>("group_topk", 2);
  router_sentinel_ = info.GetAttrOrDefault<float>("router_sentinel", -10000.0f);
  ORT_THROW_IF_ERROR(ValidateMoEGroupRouterAttributes(top_k_, n_group_, topk_group_, group_topk_));
  ORT_ENFORCE(top_k_ <= std::numeric_limits<int>::max() &&
                  n_group_ <= std::numeric_limits<int>::max() &&
                  topk_group_ <= std::numeric_limits<int>::max() &&
                  group_topk_ <= std::numeric_limits<int>::max(),
              "MoEGroupRouter CUDA attributes must not exceed INT_MAX");
}

template <typename T>
Status MoEGroupRouter<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename OrtToCudaType<T>::type CudaT;

  const Tensor* logits = context->Input<Tensor>(0);
  const Tensor* expert_bias = context->Input<Tensor>(1);
  const auto& in_shape = logits->Shape();
  ORT_RETURN_IF_NOT(in_shape.NumDimensions() >= 2, "logits must have rank >= 2");
  const int64_t num_experts = in_shape[in_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(num_experts > 0 && num_experts <= 512 && expert_bias->Shape().Size() == num_experts,
                    "expert_bias length must match num_experts (<= 512)");
  ORT_RETURN_IF_NOT(n_group_ > 0 && num_experts % n_group_ == 0 && topk_group_ > 0 &&
                        topk_group_ <= n_group_ && top_k_ <= num_experts,
                    "Invalid MoEGroupRouter group/top_k parameters");

  const int64_t num_rows = in_shape.Size() / num_experts;
  TensorShape out_shape({num_rows, num_experts});
  Tensor* output = context->Output(0, out_shape);
  if (in_shape.Size() == 0) {
    return Status::OK();
  }

  return LaunchMoEGroupRouterKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(logits->Data<T>()),
      expert_bias->Data<float>(),
      num_rows,
      static_cast<int>(num_experts),
      static_cast<int>(n_group_),
      static_cast<int>(topk_group_),
      static_cast<int>(group_topk_),
      static_cast<int>(top_k_),
      router_sentinel_);
}

template class MoEGroupRouter<float>;
template class MoEGroupRouter<MLFloat16>;
template class MoEGroupRouter<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
