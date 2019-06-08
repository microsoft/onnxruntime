// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shrink.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define SHRINK_REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Shrink,                                                                   \
      kOnnxDomain,                                                              \
      9,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

template <typename T>
Status Shrink<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>()); 

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Shrink<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)
SPECIALIZED_COMPUTE(uint8_t)
SPECIALIZED_COMPUTE(int8_t)
SPECIALIZED_COMPUTE(uint16_t)
SPECIALIZED_COMPUTE(int16_t)
SPECIALIZED_COMPUTE(uint32_t)
SPECIALIZED_COMPUTE(int32_t)
SPECIALIZED_COMPUTE(uint64_t)
SPECIALIZED_COMPUTE(int64_t)
}  // namespace cuda
}  // namespace onnxruntime
