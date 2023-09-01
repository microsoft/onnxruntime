// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "constant_of_shape.h"

namespace onnxruntime {
namespace js {

#define CONSTANT_OF_SHAPE_KERNEL_VERSIONED(domain, data_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                \
      ConstantOfShape,                                                                    \
      domain,                                                                             \
      since_version,                                                                      \
      end_version,                                                                        \
      data_type,                                                                          \
      kJsExecutionProvider,                                                               \
      (*KernelDefBuilder::Create())                                                       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())                   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<data_type>())                 \
          .InputMemoryType(OrtMemTypeCPU, 0),                                             \
      ConstantOfShape<data_type>);

#define CONSTANT_OF_SHAPE_KERNEL(domain, data_type, since_version)        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                          \
      ConstantOfShape,                                                    \
      domain,                                                             \
      since_version,                                                      \
      data_type,                                                          \
      kJsExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<data_type>()) \
          .InputMemoryType(OrtMemTypeCPU, 0),                             \
      ConstantOfShape<data_type>);

CONSTANT_OF_SHAPE_KERNEL_VERSIONED(kOnnxDomain, float, 9, 19)
CONSTANT_OF_SHAPE_KERNEL(kOnnxDomain, float, 20)
CONSTANT_OF_SHAPE_KERNEL_VERSIONED(kOnnxDomain, int32_t, 9, 19)
CONSTANT_OF_SHAPE_KERNEL(kOnnxDomain, int32_t, 20)

}  // namespace js
}  // namespace onnxruntime
