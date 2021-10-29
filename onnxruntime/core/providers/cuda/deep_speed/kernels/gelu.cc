// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gelu.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {
namespace deep_speed {

#define REGISTER_GELU_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gelu<T>);

#define REGISTER_BIAS_GELU_KERNEL_TYPED(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BiasGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BiasGelu<T>);

#define REGISTER_FAST_GELU_KERNEL_TYPED(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Gelu,                                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_GELU_KERNEL_TYPED(float)
REGISTER_BIAS_GELU_KERNEL_TYPED(float)
REGISTER_FAST_GELU_KERNEL_TYPED(float)

template <typename T>
Gelu<T>::Gelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status Gelu<T>::ComputeInternal(OpKernelContext* /*context*/) const {
  return Status::OK();
}

template <typename T>
BiasGelu<T>::BiasGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status BiasGelu<T>::ComputeInternal(OpKernelContext* /*context*/) const {
  return Status::OK();
}

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* /*context*/) const {
  return Status::OK();
}

}  // namespace deep_speed
}  // namespace cuda
}  // namespace onnxruntime
