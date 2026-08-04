// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_moe_router.h"

#include "contrib_ops/cuda/math/dsv4_moe_router_impl.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      DSV4MoERouter,                                                    \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<float>())    \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      DSV4MoERouter<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
DSV4MoERouter<T>::DSV4MoERouter(const OpKernelInfo& info) : CudaKernel(info) {
  topk_ = info.GetAttrOrDefault<int64_t>("topk", static_cast<int64_t>(1));
  local_expert_start_ = info.GetAttrOrDefault<int64_t>("local_expert_start", static_cast<int64_t>(0));
  local_expert_count_ = info.GetAttrOrDefault<int64_t>("local_expert_count", static_cast<int64_t>(0));
  route_scale_ = info.GetAttrOrDefault<float>("route_scale", 1.0f);

  // The chosen experts and the running weight sum are held in one warp.
  ORT_ENFORCE(topk_ > 0 && topk_ <= 32, "topk must be in [1, 32], got ", topk_);
  ORT_ENFORCE(local_expert_start_ >= 0, "local_expert_start must not be negative, got ",
              local_expert_start_);
  ORT_ENFORCE(local_expert_count_ > 0, "local_expert_count must be positive, got ",
              local_expert_count_);
}

template <typename T>
Status DSV4MoERouter<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* scores = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);
  const Tensor* expert_ids = context->Input<Tensor>(2);

  const auto& score_dims = scores->Shape().GetDims();
  if (score_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "scores is expected to be [tokens, num_experts], got rank ",
                           score_dims.size());
  }
  const int64_t num_tokens = score_dims[0];
  const int64_t num_experts = score_dims[1];
  if (topk_ > num_experts) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "topk ", topk_,
                           " exceeds the ", num_experts, " experts on offer.");
  }
  if (local_expert_start_ + local_expert_count_ > num_experts) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the local experts [",
                           local_expert_start_, ", ",
                           local_expert_start_ + local_expert_count_, ") run past the ",
                           num_experts, " experts on offer.");
  }
  if (bias != nullptr && bias->Shape().Size() != num_experts) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bias must hold ", num_experts,
                           " elements, got ", bias->Shape().Size());
  }
  if (expert_ids != nullptr && expert_ids->Shape() != TensorShape({num_tokens, topk_})) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "expert_ids must be [", num_tokens,
                           ", ", topk_, "], got ", expert_ids->Shape());
  }

  Tensor* router_probs = context->Output(0, TensorShape({num_tokens, local_expert_count_}));
  Tensor* weight_scale = context->Output(1, TensorShape({num_tokens, 1}));
  if (num_tokens == 0) return Status::OK();

  DSV4MoERouterParams params;
  params.num_tokens = static_cast<int>(num_tokens);
  params.num_experts = static_cast<int>(num_experts);
  params.topk = static_cast<int>(topk_);
  params.local_expert_start = static_cast<int>(local_expert_start_);
  params.local_expert_count = static_cast<int>(local_expert_count_);
  params.route_scale = route_scale_;

  return LaunchDSV4MoERouter<T>(Stream(context), params, scores->Data<float>(),
                                bias == nullptr ? nullptr : bias->Data<float>(),
                                expert_ids == nullptr ? nullptr : expert_ids->Data<int64_t>(),
                                router_probs->MutableData<T>(),
                                weight_scale->MutableData<float>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
