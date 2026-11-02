// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/router_gate_up_proj.h"

#include <type_traits>

#include "contrib_ops/cuda/math/router_gate_up_proj_impl.h"
#include "core/providers/cuda/cuda_common.h"

// Include env_var_utils.h after cuda_common.h (see hyper_connection_low_rank_mix_impl.cu).
#include "core/platform/env_var_utils.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      RouterGateUpProj,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RouterGateUpProj<T>);

REGISTER_KERNEL_TYPED(BFloat16)
REGISTER_KERNEL_TYPED(MLFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
RouterGateUpProj<T>::RouterGateUpProj(const OpKernelInfo& info) : CudaKernel(info) {
  router_size_ = info.GetAttrOrDefault<int64_t>("router_size", 0);
  ORT_ENFORCE(router_size_ > 0, "RouterGateUpProj requires a positive router_size");
  const std::string activation = info.GetAttrOrDefault<std::string>("activation", "swiglu");
  ORT_ENFORCE(activation == "swiglu", "RouterGateUpProj supports activation 'swiglu' only, got '", activation, "'");
  limit_ = info.GetAttrOrDefault<float>("limit", 0.0f);
  alpha_ = info.GetAttrOrDefault<float>("activation_alpha", 1.0f);
  beta_ = info.GetAttrOrDefault<float>("activation_beta", 0.0f);
  // The fused kernel can prefetch W before it waits for the producer of X, which is only valid for a constant W.
  const Tensor* constant_input = nullptr;
  weight_is_constant_ = info.TryGetConstantInput(1, &constant_input);
}

template <typename T>
Status RouterGateUpProj<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* x = context->Input<Tensor>(0);
  const Tensor* w = context->Input<Tensor>(1);
  const TensorShape& x_shape = x->Shape();
  const TensorShape& w_shape = w->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(w_shape.NumDimensions() == 2, "W must have rank 2");
  const size_t last_axis = x_shape.NumDimensions() - 1;
  const int64_t k = x_shape[last_axis];
  const int64_t n = w_shape[1];
  const int64_t rest = n - router_size_;
  ORT_RETURN_IF_NOT(w_shape[0] == k, "W's first dimension must match X's last dimension");
  ORT_RETURN_IF_NOT(rest > 0 && rest % 2 == 0,
                    "W's second dimension must be router_size plus an even positive number");
  const int64_t inter = rest / 2;
  const int64_t rows = x_shape.SizeToDimension(last_axis);

  TensorShapeVector router_dims = x_shape.AsShapeVector();
  router_dims[last_axis] = router_size_;
  TensorShapeVector act_dims = x_shape.AsShapeVector();
  act_dims[last_axis] = inter;
  Tensor* router = context->Output(0, TensorShape(router_dims));
  Tensor* act = context->Output(1, TensorShape(act_dims));
  if (rows == 0) return Status::OK();

  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  if (RouterGateUpProjUsesFusedKernel(GetDeviceProp().major, rows, k, router_size_, inter, x->DataRaw(),
                                      w->DataRaw())) {
    // Programmatic dependent launch is opt-in: with it the early-resident CTAs slow the producer kernel down more
    // than the overlapped weight prefetch gains (about +0.1 ms per decode cycle on H200).
    static const bool pdl_enabled =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ROUTER_GATE_UP_PDL", 0) != 0;
    return LaunchRouterGateUpProjFused(Stream(context), weight_is_constant_ && pdl_enabled, is_bf16, x->DataRaw(),
                                       w->DataRaw(), router->MutableDataRaw(), act->MutableDataRaw(), rows, k, n,
                                       router_size_, inter, limit_, alpha_, beta_);
  }

  auto scratch = GetScratchBuffer<T>(static_cast<size_t>(rows * n), GetComputeStream(context));
  return LaunchRouterGateUpProjGeneric(GetCublasHandle(context), Stream(context), is_bf16, x->DataRaw(), w->DataRaw(),
                                       scratch.get(), router->MutableDataRaw(), act->MutableDataRaw(), rows, k, n,
                                       router_size_, inter, limit_, alpha_, beta_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
