// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/scale.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_SCALE_KERNEL_TYPED(T)                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Scale,                                                                    \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
          .TypeConstraint("ScaleT", {DataTypeImpl::GetTensorType<float>(),      \
                                     DataTypeImpl::GetTensorType<double>(),     \
                                     DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                     DataTypeImpl::GetTensorType<int64_t>(),    \
                                     DataTypeImpl::GetTensorType<int32_t>()})   \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                              \
      Scale<T>);

template <typename T>
Scale<T>::Scale(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t scale_down;
  info.GetAttrOrDefault("scale_down", &scale_down, static_cast<int64_t>(0));
  scale_down_ = (scale_down != 0);
}

template <typename T>
Status Scale<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  float scale_value;
  auto scale_tensor = context->Input<Tensor>(1);
  utils::MLTypeCallDispatcher<float, double, MLFloat16, int64_t, int32_t> t_disp(scale_tensor->GetElementType());
  t_disp.Invoke<GetScaleValueImpl>(scale_tensor, scale_value);

  if (scale_down_) {
    scale_value = 1.0f / scale_value;
  }

  auto lhs_tensor = context->Input<Tensor>(0);
  auto output_tensor = context->Output(0, lhs_tensor->Shape());
  Impl_Scale<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(lhs_tensor->template Data<T>()),
      scale_value,
      reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>()),
      output_tensor->Shape().Size());

  return Status::OK();
}

REGISTER_SCALE_KERNEL_TYPED(MLFloat16)
REGISTER_SCALE_KERNEL_TYPED(float)
REGISTER_SCALE_KERNEL_TYPED(double)

template Status Scale<MLFloat16>::ComputeInternal(OpKernelContext* context) const;
template Status Scale<float>::ComputeInternal(OpKernelContext* context) const;
template Status Scale<double>::ComputeInternal(OpKernelContext* context) const;

}  // namespace cuda
}  // namespace onnxruntime
