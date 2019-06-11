// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shrink.h"
#include "shrink_impl.h"
#include "core/providers/common.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define SHRINK_REGISTER_KERNEL(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Shrink,                                                     \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Shrink<T>);

template <typename T>
Status Shrink<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const T* x_data = X->Data<T>();
  const TensorShape& x_shape = X->Shape();
  const size_t x_size = static_cast<size_t>(x_shape.Size());

  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  auto* y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  ShrinkImpl<T>(x_data, *(reinterpret_cast<const ToCudaType<float>::MappedType*>(&bias_)), 
               *(reinterpret_cast<ToCudaType<float>::MappedType*>(&lambd_)), y_data, x_size);

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  SHRINK_REGISTER_KERNEL(T)    \
  template Status Shrink<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
//SPECIALIZED_COMPUTE(MLFloat16)
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