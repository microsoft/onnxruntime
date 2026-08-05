// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/hyper_connection_mix.h"

#include "contrib_ops/cuda/math/hyper_connection_mix_impl.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool HyperConnectionFinishFastDisabled() {
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_HC_FINISH_FAST", 0) == 1;
  return disabled;
}

bool HyperConnectionFinishVecDisabled() {
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_HC_FINISH_VEC", 0) == 1;
  return disabled;
}

bool HyperConnectionPartialGroupsEnabled() {
  static const bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_HC_PARTIAL_GROUPS", 0) == 1;
  return enabled;
}

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      HyperConnectionMix,                                             \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<float>()), \
      HyperConnectionMix<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
HyperConnectionMix<T>::HyperConnectionMix(const OpKernelInfo& info) : CudaKernel(info) {
  sinkhorn_iterations_ =
      static_cast<int>(info.GetAttrOrDefault<int64_t>("sinkhorn_iterations", static_cast<int64_t>(1)));
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
  hc_epsilon_ = info.GetAttrOrDefault<float>("hc_epsilon", 1e-6f);
  sinkhorn_epsilon_ = info.GetAttrOrDefault<float>("sinkhorn_epsilon", 1e-6f);
  post_alpha_ = info.GetAttrOrDefault<float>("post_alpha", 2.0f);
  ORT_ENFORCE(sinkhorn_iterations_ >= 1, "sinkhorn_iterations must be at least 1, got ",
              sinkhorn_iterations_);
}

template <typename T>
Status HyperConnectionMix<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* x = context->Input<Tensor>(0);
  const Tensor* residual = context->Input<Tensor>(1);
  const Tensor* post_mix = context->Input<Tensor>(2);
  const Tensor* comb_mix = context->Input<Tensor>(3);
  const Tensor* fn = context->Input<Tensor>(4);
  const Tensor* scale = context->Input<Tensor>(5);
  const Tensor* base = context->Input<Tensor>(6);
  const Tensor* norm_weight = context->Input<Tensor>(7);

  const auto& residual_dims = residual->Shape().GetDims();
  if (residual_dims.size() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "residual is expected to have rank at least 3 (..., hc, dim), got ",
                           residual_dims.size());
  }

  const int64_t dim = residual_dims[residual_dims.size() - 1];
  const int64_t hc = residual_dims[residual_dims.size() - 2];
  const int64_t num_tokens = residual->Shape().Size() / std::max<int64_t>(hc * dim, 1);
  const int64_t mix_dim = (2 + hc) * hc;

  if (hc < 1 || hc > kHyperConnectionMaxMult) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The hyper-connection multiplicity must be in [1, ",
                           kHyperConnectionMaxMult, "], got ", hc);
  }
  if (x->Shape().Size() != num_tokens * dim) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "x must hold ", num_tokens * dim, " elements to match residual, got ",
                           x->Shape().Size());
  }
  if (post_mix->Shape().Size() != num_tokens * hc || comb_mix->Shape().Size() != num_tokens * hc * hc) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "post_mix and comb_mix must be shaped (..., hc) and (..., hc, hc).");
  }
  if (fn->Shape().Size() != hc * dim * mix_dim) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fn must be shaped (hc * dim, ", mix_dim,
                           "), which is ", hc * dim * mix_dim, " elements, got ",
                           fn->Shape().Size());
  }
  if (scale->Shape().Size() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "scale must hold at least 3 elements.");
  }
  if (base->Shape().Size() != mix_dim) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "base must hold ", mix_dim,
                           " elements, got ", base->Shape().Size());
  }
  if (norm_weight->Shape().Size() != dim) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "norm_weight must hold ", dim,
                           " elements, got ", norm_weight->Shape().Size());
  }

  Tensor* residual_out = context->Output(0, residual->Shape());
  Tensor* post_mix_out = context->Output(1, post_mix->Shape());
  Tensor* comb_mix_out = context->Output(2, comb_mix->Shape());
  Tensor* layer_input = context->Output(3, x->Shape());

  HyperConnectionMixParams params;
  params.num_tokens = static_cast<int>(num_tokens);
  params.hc = static_cast<int>(hc);
  params.dim = static_cast<int>(dim);
  params.mix_dim = static_cast<int>(mix_dim);
  params.sinkhorn_iterations = sinkhorn_iterations_;
  params.epsilon = epsilon_;
  params.hc_epsilon = hc_epsilon_;
  params.sinkhorn_epsilon = sinkhorn_epsilon_;
  params.post_alpha = post_alpha_;

  auto workspace = GetScratchBuffer<float>(
      HyperConnectionMixWorkspaceFloats(params.num_tokens, params.hc, params.dim),
      context->GetComputeStream());

  return LaunchHyperConnectionMix<T>(Stream(context), params,
                                     x->Data<T>(),
                                     residual->Data<T>(),
                                     post_mix->Data<float>(),
                                     comb_mix->Data<float>(),
                                     fn->Data<float>(),
                                     scale->Data<float>(),
                                     base->Data<float>(),
                                     norm_weight->Data<float>(),
                                     workspace.get(),
                                     residual_out->MutableData<T>(),
                                     post_mix_out->MutableData<float>(),
                                     comb_mix_out->MutableData<float>(),
                                     layer_input->MutableData<T>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
