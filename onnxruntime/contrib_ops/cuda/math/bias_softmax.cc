// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      BiasSoftmax,                                                              \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BiasSoftmax<T>);                                                              

template <typename T>
Status BiasSoftmax<T>::ComputeInternal(OpKernelContext* ctx) const {

  typedef typename ToCudaType<T>::MappedType CudaT;

  auto X_data = reinterpret_cast<const CudaT*>(ctx->Input<Tensor>(0)->template Data<T>());
  const TensorShape& X_shape{ctx->Input<Tensor>(0)->Shape()};
  auto B_data = reinterpret_cast<const CudaT*>(ctx->Input<Tensor>(1)->template Data<T>());
  const TensorShape& B_shape{ctx->Input<Tensor>(1)->Shape()};
  auto Y_data = reinterpret_cast<CudaT*>(ctx->Output(0, X_shape)->template MutableData<T>());

  const int64_t softmax_axis = HandleNegativeAxis(softmax_axis_, X_shape.NumDimensions());
  int N = (int)X_shape.SizeToDimension(softmax_axis);
  int D = (int)X_shape.SizeFromDimension(softmax_axis);

  const int64_t broadcast_axis = HandleNegativeAxis(broadcast_axis_, X_shape.NumDimensions());
  int broadcast_size = N/(int)X_shape.SizeToDimension(broadcast_axis);

  if (D <= 1024 && D*sizeof(T) <= 4096) {

    // expect thread blocks can fill SM at high occupancy without overflowing registers
    dispatch_bias_softmax_forward<CudaT, CudaT, AccType<T>>(Y_data, X_data, B_data, D, N, D, broadcast_size);
  }
  else {

    // need to fallback to add kernel + CUDA DNN library softmax call :/
    dispatch_bias_softmax_softward_via_dnn_library<CudaT>(
      CudnnHandle(), D, N, broadcast_axis, softmax_axis, X_shape, X_data, B_shape, B_data, Y_data);
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status BiasSoftmax<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
