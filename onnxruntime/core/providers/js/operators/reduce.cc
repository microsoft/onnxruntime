// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduce.h"

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(x, sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                         \
      ReduceMean,                                                                  \
      kOnnxDomain,                                                                 \
      sinceVersion, endVersion,                                                    \
      float,                                                                       \
      kJsExecutionProvider,                                                        \
      (*KernelDefBuilder::Create())                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),              \
      ReduceMean<float>);                                                          \
                                                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                         \
      ReduceMean,                                                                  \
      kOnnxDomain,                                                                 \
      sinceVersion, endVersion,                                                    \
      int32_t,                                                                     \
      kJsExecutionProvider,                                                        \
      (*KernelDefBuilder::Create())                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),            \
      ReduceMean<float>);

#define REGISTER_REDUCE_ELEMENTWISE_KERNEL(x, sinceVersion)             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      ReduceMean,                                                       \
      kOnnxDomain,                                                      \
      sinceVersion,                                                     \
      float,                                                            \
      kJsExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),   \
      ReduceMean<float>);                                               \
                                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      ReduceMean,                                                       \
      kOnnxDomain,                                                      \
      sinceVersion,                                                               \
      int32_t,                                                          \
      kJsExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), \
      ReduceMean<float>);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMean, 1, 10)
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMean, 11, 12)
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMean, 13)

}  // namespace js
}  // namespace onnxruntime
