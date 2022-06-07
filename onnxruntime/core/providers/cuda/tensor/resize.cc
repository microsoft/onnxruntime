// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T)                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      10, 10,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      Resize<T>);                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      11, 12,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      13,                                                          \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(int32_t)
REGISTER_KERNEL_TYPED(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime
