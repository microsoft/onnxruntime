// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace js {
#define REGISTER_RESIZE_VERSIONED_10_10_KERNEL(domain)     \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                       \
      Resize,                                              \
      domain,                                              \
      10, 10,                                              \
      kJsExecutionProvider,                                \
      (*KernelDefBuilder::Create())                        \
          .InputMemoryType(OrtMemTypeCPUInput, 1)          \
          .TypeConstraint("T", JsepSupportedFloatTypes()), \
      Resize);

#define REGISTER_RESIZE_VERSIONED_KERNEL(domain, sinceVersion, endVerion) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                      \
      Resize,                                                             \
      domain,                                                             \
      sinceVersion, endVerion,                                            \
      kJsExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                       \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                         \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                         \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                         \
          .TypeConstraint("T1", JsepSupportedFloatTypes())                \
          .TypeConstraint("T2", JsepSupportedFloatTypes()),               \
      Resize);

#define REGISTER_RESIZE_KERNEL(domain, sinceVersion)        \
  ONNX_OPERATOR_KERNEL_EX(                                  \
      Resize,                                               \
      domain,                                               \
      sinceVersion,                                         \
      kJsExecutionProvider,                                 \
      (*KernelDefBuilder::Create())                         \
          .InputMemoryType(OrtMemTypeCPUInput, 1)           \
          .InputMemoryType(OrtMemTypeCPUInput, 2)           \
          .InputMemoryType(OrtMemTypeCPUInput, 3)           \
          .TypeConstraint("T1", JsepSupportedFloatTypes())  \
          .TypeConstraint("T2", JsepSupportedFloatTypes()), \
      Resize);

#define REGISTER_RESIZE_KERNEL_DOMAIN(domain)       \
  REGISTER_RESIZE_VERSIONED_KERNEL(domain, 11, 12); \
  REGISTER_RESIZE_VERSIONED_KERNEL(domain, 13, 17); \
  REGISTER_RESIZE_VERSIONED_KERNEL(domain, 18, 18); \
  REGISTER_RESIZE_KERNEL(domain, 19);

REGISTER_RESIZE_VERSIONED_10_10_KERNEL(kOnnxDomain);
REGISTER_RESIZE_VERSIONED_10_10_KERNEL(kMSInternalNHWCDomain);
REGISTER_RESIZE_KERNEL_DOMAIN(kOnnxDomain);
REGISTER_RESIZE_KERNEL_DOMAIN(kMSInternalNHWCDomain);

}  // namespace js
}  // namespace onnxruntime
