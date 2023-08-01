// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "pad.h"

namespace onnxruntime {
namespace js {

#define PAD_KERNEL_VERSIONED_WITH_MODE_PADS_VALUE_ATTRIBUTES(domain, data_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                  \
      Pad,                                                                                                  \
      domain,                                                                                               \
      since_version,                                                                                        \
      end_version,                                                                                          \
      data_type,                                                                                            \
      kJsExecutionProvider,                                                                                 \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),          \
      Pad<data_type>);

#define PAD_KERNEL_VERSIONED_WITH_MODE_ATTRIBUTE(domain, data_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                      \
      Pad,                                                                                      \
      domain,                                                                                   \
      since_version,                                                                            \
      end_version,                                                                              \
      data_type,                                                                                \
      kJsExecutionProvider,                                                                     \
      (*KernelDefBuilder::Create())                                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())                        \
          .InputMemoryType(OrtMemTypeCPU, 1)                                                    \
          .InputMemoryType(OrtMemTypeCPU, 2)                                                    \
          .InputMemoryType(OrtMemTypeCPU, 3),                                                   \
      Pad<data_type>);

#define PAD_KERNEL_WITH_MODE_ATTRIBUTE(domain, data_type, since_version) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      Pad,                                                               \
      domain,                                                            \
      since_version,                                                     \
      data_type,                                                         \
      kJsExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()) \
          .InputMemoryType(OrtMemTypeCPU, 1)                             \
          .InputMemoryType(OrtMemTypeCPU, 2)                             \
          .InputMemoryType(OrtMemTypeCPU, 3),                            \
      Pad<data_type>);

PAD_KERNEL_VERSIONED_WITH_MODE_PADS_VALUE_ATTRIBUTES(kOnnxDomain, float, 1, 1)
PAD_KERNEL_VERSIONED_WITH_MODE_PADS_VALUE_ATTRIBUTES(kOnnxDomain, float, 2, 10)
PAD_KERNEL_VERSIONED_WITH_MODE_ATTRIBUTE(kOnnxDomain, float, 11, 12)
PAD_KERNEL_VERSIONED_WITH_MODE_ATTRIBUTE(kOnnxDomain, float, 13, 17)
PAD_KERNEL_VERSIONED_WITH_MODE_ATTRIBUTE(kOnnxDomain, float, 18, 18)
PAD_KERNEL_WITH_MODE_ATTRIBUTE(kOnnxDomain, float, 19)

}  // namespace js
}  // namespace onnxruntime
