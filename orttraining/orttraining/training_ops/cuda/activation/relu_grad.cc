// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/relu_grad.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_RELU_GRAD_KERNEL(x, ver, domain, T)             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      domain,                                                    \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                         \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    return Status::OK();                                                                                         \
  }

#define RELU_GRAD_OP_TYPED(name, ver, domain, T)  \
  REGISTER_RELU_GRAD_KERNEL(name, ver, domain, T) 

#define RELU_GRAD_OP_HFD(name, ver, domain)        \
  RELU_GRAD_OP_TYPED(name, ver, domain, MLFloat16) \
  RELU_GRAD_OP_TYPED(name, ver, domain, float)     \
  RELU_GRAD_OP_TYPED(name, ver, domain, double)

RELU_GRAD_OP_HFD(ReluGrad, 1, kMSDomain);

}  //namespace cuda
}  // namespace onnxruntime
