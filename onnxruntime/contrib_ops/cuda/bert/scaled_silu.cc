// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/scaled_silu.h"

#include "contrib_ops/cuda/bert/scaled_silu_impl.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

#define REGISTER_SCALED_SILU_CUDA_KERNEL(T)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      ScaledSiLU, kMSDomain, 1, T, kCudaExecutionProvider,                 \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      ScaledSiLU<T>);

REGISTER_SCALED_SILU_CUDA_KERNEL(float)
REGISTER_SCALED_SILU_CUDA_KERNEL(MLFloat16)
REGISTER_SCALED_SILU_CUDA_KERNEL(BFloat16)

#undef REGISTER_SCALED_SILU_CUDA_KERNEL

template <typename T>
ScaledSiLU<T>::ScaledSiLU(const OpKernelInfo& info)
    : CudaKernel(info), alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

template <typename T>
Status ScaledSiLU<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0, "scale must be a scalar");
  auto* y = context->Output(0, x->Shape());
  return LaunchScaledSiLU<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(x->Data<T>()),
      scale == nullptr ? nullptr : scale->DataRaw(), scale == nullptr ? 0 : scale->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()), x->Shape().Size(), alpha_);
}

template class ScaledSiLU<float>;
template class ScaledSiLU<MLFloat16>;
template class ScaledSiLU<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
