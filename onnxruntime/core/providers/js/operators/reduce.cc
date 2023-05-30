// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduce.h"

namespace onnxruntime {
namespace js {

#define REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceOp, sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                \
      ReduceOp,                                                                           \
      kOnnxDomain,                                                                        \
      sinceVersion, endVersion,                                                           \
      float,                                                                              \
      kJsExecutionProvider,                                                               \
      (*KernelDefBuilder::Create())                                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),                     \
      ReduceOp<float>);                                                                   \
                                                                                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                \
      ReduceOp,                                                                           \
      kOnnxDomain,                                                                        \
      sinceVersion, endVersion,                                                           \
      int32_t,                                                                            \
      kJsExecutionProvider,                                                               \
      (*KernelDefBuilder::Create())                                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),                   \
      ReduceOp<int32_t>);

#define REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceOp, sinceVersion)         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      ReduceOp,                                                            \
      kOnnxDomain,                                                         \
      sinceVersion,                                                        \
      float,                                                               \
      kJsExecutionProvider,                                                \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())       \
          .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>()), \
      ReduceOp<float>);                                                    \
                                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      ReduceOp,                                                            \
      kOnnxDomain,                                                         \
      sinceVersion,                                                        \
      int32_t,                                                             \
      kJsExecutionProvider,                                                \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())     \
          .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>()), \
      ReduceOp<int32_t>);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMean, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMean, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMean, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMean, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMax, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMax, 11, 11);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMax, 12, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMax, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMax, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMin, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMin, 11, 11);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMin, 12, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceMin, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMin, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceProd, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceProd, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceProd, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceProd, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceSum, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceSum, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceSum, 13);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceLogSum, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceLogSum, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceLogSum, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceLogSum, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceSumSquare, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceSumSquare, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceSumSquare, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceSumSquare, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceLogSumExp, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceLogSumExp, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONSED_KERNEL(ReduceLogSumExp, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceLogSumExp, 18);

}  // namespace js
}  // namespace onnxruntime
