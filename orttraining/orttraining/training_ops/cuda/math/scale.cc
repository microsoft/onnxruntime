// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scale.h"

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
                                     DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                     DataTypeImpl::GetTensorType<int64_t>()})   \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                              \
      Scale<T>);

template <typename T>
Status Scale<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  float scale_value;
  auto scale_tensor = context->Input<Tensor>(1);
  utils::MLTypeCallDispatcher<GetScaleValueImpl, float, MLFloat16, int64_t> t_disp(scale_tensor->GetElementType());
  t_disp.Invoke(scale_tensor, scale_value);
  float inverse_scale_value = 1.0f / scale_value;

  auto lhs_tensor = context->Input<Tensor>(0);
  auto output_tensor = context->Output(0, lhs_tensor->Shape());
  Impl_Scale<CudaT>(
      reinterpret_cast<const CudaT*>(lhs_tensor->template Data<T>()),
      inverse_scale_value,
      reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>()),
      output_tensor->Shape().Size());

  return Status::OK();
}

REGISTER_SCALE_KERNEL_TYPED(MLFloat16)
REGISTER_SCALE_KERNEL_TYPED(float)

template Status Scale<MLFloat16>::ComputeInternal(OpKernelContext* context) const;
template Status Scale<float>::ComputeInternal(OpKernelContext* context) const;

}  // namespace cuda
}  // namespace onnxruntime
