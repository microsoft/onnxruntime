// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/branchwise_rms_norm.h"

#include <limits>

#include "contrib_ops/cuda/bert/branchwise_rms_norm_impl.h"
#include "contrib_ops/hyper_connection_helper.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

using namespace onnxruntime::cuda;

#define REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      BranchwiseRMSNorm, kMSDomain, 1, T, kCudaExecutionProvider,          \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      BranchwiseRMSNorm<T>);

REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(float)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(MLFloat16)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(BFloat16)

#undef REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL

template <typename T>
BranchwiseRMSNorm<T>::BranchwiseRMSNorm(const OpKernelInfo& info)
    : CudaKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status BranchwiseRMSNorm<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(x->Shape(), num_branches_, params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
  }
  ORT_RETURN_IF_NOT(params.branches <= std::numeric_limits<int>::max() &&
                        params.hidden <= std::numeric_limits<int>::max(),
                    "branch or hidden dimension is too large for CUDA");

  const int64_t groups = x->Shape().Size() / params.hidden;
  auto* y = context->Output(0, x->Shape());
  if (groups == 0) {
    return Status::OK();
  }

  const auto* x_data = reinterpret_cast<const CudaT*>(x->Data<T>());
  auto* y_data = reinterpret_cast<CudaT*>(y->MutableData<T>());
  return LaunchMixedScaleBranchwiseRMSNorm<CudaT>(
      Stream(context), x_data, scale == nullptr ? nullptr : scale->DataRaw(),
      scale == nullptr ? x->GetElementType() : scale->GetElementType(), y_data,
      groups, static_cast<int>(params.branches), static_cast<int>(params.hidden),
      scale == nullptr || scale->Shape().Size() == params.hidden, epsilon_);
}

template class BranchwiseRMSNorm<float>;
template class BranchwiseRMSNorm<MLFloat16>;
template class BranchwiseRMSNorm<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
