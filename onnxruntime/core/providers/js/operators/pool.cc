// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "pool.h"

namespace onnxruntime {
namespace js {

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                   \
      op_name,                                                                                     \
      kOnnxDomain,                                                                                 \
      since_version,                                                                               \
      data_type,                                                                                   \
      kJsExecutionProvider,                                                                        \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL_VERSIONED(op_name, data_type, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                  \
      op_name,                                                                              \
      kOnnxDomain,                                                                          \
      since_version,                                                                        \
      end_version,                                                                          \
      data_type,                                                                            \
      kJsExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create())                                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),                   \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL_WITH_INDICES(op_name, data_type, pool_type, since_version) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                  \
      op_name,                                                                    \
      kOnnxDomain,                                                                \
      since_version,                                                              \
      data_type,                                                                  \
      kJsExecutionProvider,                                                       \
      (*KernelDefBuilder::Create())                                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())          \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),           \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL_VERSIONED_WITH_INDICES(op_name, data_type, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                               \
      op_name,                                                                                           \
      kOnnxDomain,                                                                                       \
      since_version,                                                                                     \
      end_version,                                                                                       \
      data_type,                                                                                         \
      kJsExecutionProvider,                                                                              \
      (*KernelDefBuilder::Create())                                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())                                 \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),                                  \
      Pool<data_type, pool_type>);

POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10)
POOLING_KERNEL(AveragePool, float, AveragePool, 11)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1)

POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 11, 11)
POOLING_KERNEL_WITH_INDICES(MaxPool, float, MaxPool<8>, 12)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1)

}  // namespace js
}  // namespace onnxruntime
