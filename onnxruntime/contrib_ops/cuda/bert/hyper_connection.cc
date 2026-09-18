// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection.h"

#include <limits>

#include "contrib_ops/cuda/bert/hyper_connection_impl.h"
#include "contrib_ops/hyper_connection_helper.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

using hyper_connection::GateLayout;
using hyper_connection::StreamShape;
using namespace onnxruntime::cuda;

#define REGISTER_HC_CUDA_KERNEL(Op, T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Op, kMSDomain, 1, T, kCudaExecutionProvider,                \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(), \
                                 DataTypeImpl::GetTensorType<MLFloat16>(), \
                                 DataTypeImpl::GetTensorType<BFloat16>()}), \
      Op<T>);

#define REGISTER_HC_CUDA_TYPES(Op)       \
  REGISTER_HC_CUDA_KERNEL(Op, float)     \
  REGISTER_HC_CUDA_KERNEL(Op, MLFloat16) \
  REGISTER_HC_CUDA_KERNEL(Op, BFloat16)

REGISTER_HC_CUDA_TYPES(BranchwiseRMSNorm)
REGISTER_HC_CUDA_TYPES(ScaledSiLU)
REGISTER_HC_CUDA_TYPES(HyperConnectionPreMix)
REGISTER_HC_CUDA_TYPES(HyperConnectionPostMix)

#undef REGISTER_HC_CUDA_TYPES
#undef REGISTER_HC_CUDA_KERNEL

namespace {

Status ValidateCudaParams(const StreamShape& params) {
  ORT_RETURN_IF_NOT(params.branches <= std::numeric_limits<int>::max() &&
                        params.hidden <= std::numeric_limits<int>::max(),
                    "branch or hidden dimension is too large for CUDA");
  return Status::OK();
}

}  // namespace

template <typename T>
BranchwiseRMSNorm<T>::BranchwiseRMSNorm(const OpKernelInfo& info)
    : CudaKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status BranchwiseRMSNorm<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      x->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(ValidateCudaParams(params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
  }
  auto* y = context->Output(0, x->Shape());
  return LaunchBranchwiseRMSNorm<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(x->Data<T>()),
      scale == nullptr ? nullptr : scale->DataRaw(),
      scale == nullptr ? 0 : scale->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()),
      x->Shape().Size() / params.hidden, static_cast<int>(params.branches),
      static_cast<int>(params.hidden),
      scale != nullptr && scale->Shape().Size() == params.hidden, epsilon_);
}

template <typename T>
ScaledSiLU<T>::ScaledSiLU(const OpKernelInfo& info)
    : CudaKernel(info), alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

template <typename T>
Status ScaledSiLU<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0,
                    "scale must be a scalar");
  auto* y = context->Output(0, x->Shape());
  return LaunchScaledSiLU<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(x->Data<T>()),
      scale == nullptr ? nullptr : scale->DataRaw(),
      scale == nullptr ? 0 : scale->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()), x->Shape().Size(), alpha_);
}

template <typename T>
HyperConnectionPreMix<T>::HyperConnectionPreMix(const OpKernelInfo& info)
    : CudaKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      reduction_scale_(
          info.GetAttrOrDefault<float>("reduction_scale", 1.0f)) {}

template <typename T>
Status HyperConnectionPreMix<T>::ComputeInternal(
    OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const auto* streams = context->Input<Tensor>(0);
  const auto* pre_mix = context->Input<Tensor>(1);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(ValidateCudaParams(params));
  GateLayout layout;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
      pre_mix->Shape(), streams->Shape(), params, false, layout));
  auto* y = context->Output(0, TensorShape(params.reduced_shape));
  return LaunchHyperConnectionPreMix<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(streams->Data<T>()),
      pre_mix->DataRaw(), pre_mix->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()), y->Shape().Size(),
      static_cast<int>(params.branches), static_cast<int>(params.hidden),
      static_cast<int>(layout), reduction_scale_);
}

template <typename T>
HyperConnectionPostMix<T>::HyperConnectionPostMix(const OpKernelInfo& info)
    : CudaKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status HyperConnectionPostMix<T>::ComputeInternal(
    OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const auto* streams = context->Input<Tensor>(0);
  const auto* branch_output = context->Input<Tensor>(1);
  const auto* post_mix = context->Input<Tensor>(2);
  const auto* stream_mix = context->Input<Tensor>(3);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(ValidateCudaParams(params));
  ORT_RETURN_IF_ERROR(
      hyper_connection::ValidateReduced(branch_output->Shape(), params));
  GateLayout layout;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
      post_mix->Shape(), streams->Shape(), params, false, layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(
        stream_mix->Shape(), streams->Shape(), params));
  }
  auto* y = context->Output(0, streams->Shape());
  return LaunchHyperConnectionPostMix<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(streams->Data<T>()),
      reinterpret_cast<const CudaT*>(branch_output->Data<T>()),
      post_mix->DataRaw(),
      stream_mix == nullptr ? nullptr : stream_mix->DataRaw(),
      post_mix->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()), streams->Shape().Size(),
      static_cast<int>(params.branches), static_cast<int>(params.hidden),
      static_cast<int>(layout));
}

#define INSTANTIATE(Op)         \
  template class Op<float>;     \
  template class Op<MLFloat16>; \
  template class Op<BFloat16>;

INSTANTIATE(BranchwiseRMSNorm)
INSTANTIATE(ScaledSiLU)
INSTANTIATE(HyperConnectionPreMix)
INSTANTIATE(HyperConnectionPostMix)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
