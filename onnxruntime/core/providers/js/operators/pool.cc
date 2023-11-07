// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "pool.h"

namespace onnxruntime {
namespace js {

#define POOLING_KERNEL(op_name, domain, is_channels_last, pool_type, since_version) \
  ONNX_OPERATOR_KERNEL_EX(                                                          \
      op_name,                                                                      \
      domain,                                                                       \
      since_version,                                                                \
      kJsExecutionProvider,                                                         \
      (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()), \
      Pool<pool_type, is_channels_last>);

#define POOLING_KERNEL_VERSIONED(op_name, domain, is_channels_last, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                       \
      op_name,                                                                                             \
      domain,                                                                                              \
      since_version,                                                                                       \
      end_version,                                                                                         \
      kJsExecutionProvider,                                                                                \
      (*KernelDefBuilder::Create())                                                                        \
          .TypeConstraint("T", JsepSupportedFloatTypes()),                                                 \
      Pool<pool_type, is_channels_last>);

#define POOLING_KERNEL_WITH_INDICES(op_name, domain, is_channels_last, pool_type, since_version) \
  ONNX_OPERATOR_KERNEL_EX(                                                                       \
      op_name,                                                                                   \
      domain,                                                                                    \
      since_version,                                                                             \
      kJsExecutionProvider,                                                                      \
      (*KernelDefBuilder::Create())                                                              \
          .TypeConstraint("T", JsepSupportedFloatTypes())                                        \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),                          \
      Pool<pool_type, is_channels_last>);

#define POOLING_KERNEL_VERSIONED_WITH_INDICES(op_name, domain, is_channels_last, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                                    \
      op_name,                                                                                                          \
      domain,                                                                                                           \
      since_version,                                                                                                    \
      end_version,                                                                                                      \
      kJsExecutionProvider,                                                                                             \
      (*KernelDefBuilder::Create())                                                                                     \
          .TypeConstraint("T", JsepSupportedFloatTypes())                                                               \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),                                                 \
      Pool<pool_type, is_channels_last>);

POOLING_KERNEL_VERSIONED(AveragePool, kOnnxDomain, false, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, kMSInternalNHWCDomain, true, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, kOnnxDomain, false, AveragePool, 10, 10)
POOLING_KERNEL_VERSIONED(AveragePool, kMSInternalNHWCDomain, true, AveragePool, 10, 10)
POOLING_KERNEL(AveragePool, kOnnxDomain, false, AveragePool, 11)
POOLING_KERNEL(AveragePool, kMSInternalNHWCDomain, true, AveragePool, 11)
POOLING_KERNEL(GlobalAveragePool, kOnnxDomain, false, AveragePool, 1)
POOLING_KERNEL(GlobalAveragePool, kMSInternalNHWCDomain, true, AveragePool, 1)

POOLING_KERNEL_VERSIONED(MaxPool, kOnnxDomain, false, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, kMSInternalNHWCDomain, true, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 11, 11)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 11, 11)
POOLING_KERNEL_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 12)
POOLING_KERNEL_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 12)
POOLING_KERNEL(GlobalMaxPool, kOnnxDomain, false, MaxPool<1>, 1)
POOLING_KERNEL(GlobalMaxPool, kMSInternalNHWCDomain, true, MaxPool<1>, 1)

}  // namespace js
}  // namespace onnxruntime
