// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace js {
#define REGISTER_RESIZE_ELEMENTWISE_VERSIONED_KERNEL(sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                           \
      Resize,                                                                  \
      kOnnxDomain,                                                             \
      sinceVersion, endVersion,                                                \
      kJsExecutionProvider,                                                    \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())          \
          .InputMemoryType(OrtMemTypeCPU, 1)                                   \
              Resize);                                                         \
                                                                               \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                           \
      Resize,                                                                  \
      kOnnxDomain,                                                             \
      sinceVersion, endVersion,                                                \
      kJsExecutionProvider,                                                    \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>())          \
          .InputMemoryType(OrtMemTypeCPU, 1)                                   \
          .InputMemoryType(OrtMemTypeCPU, 2)                                   \
              Resize);

ONNX_OPERATOR_KERNEL_EX(
    Resize,
    kOnnxDomain,
    19,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1)  // scales or size
    Resize);

ONNX_OPERATOR_KERNEL_EX(
    Resize,
    kOnnxDomain,
    19,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1)  // roi
        .InputMemoryType(OrtMemTypeCPU, 2)  // scales or size
    Resize);

REGISTER_RESIZE_ELEMENTWISE_VERSIONED_KERNEL(10, 10);
REGISTER_RESIZE_ELEMENTWISE_VERSIONED_KERNEL(11, 12);
REGISTER_RESIZE_ELEMENTWISE_VERSIONED_KERNEL(13, 17);
REGISTER_RESIZE_ELEMENTWISE_VERSIONED_KERNEL(18, 18);

}  // namespace js
}  // namespace onnxruntime
