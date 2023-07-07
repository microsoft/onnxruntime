// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reduce.h"

namespace onnxruntime {
namespace js {

#define REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceOp, sinceVersion, endVersion) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                               \
      ReduceOp,                                                                          \
      kOnnxDomain,                                                                       \
      sinceVersion, endVersion,                                                          \
      float,                                                                             \
      kJsExecutionProvider,                                                              \
      (*KernelDefBuilder::Create())                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),                    \
      ReduceOp<float>);

// macro REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL does not set .InputMemoryType(OrtMemTypeCPU, 1), so in future if
// a new opset version update applies to Reduce* operators, we may need to add another macro like
// REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL_WITH_AXIS_IN_INPUT to set input memory type.
// i.e. we cannot use REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL to version 18 when the opset version is increased.

#define REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceOp, sinceVersion)   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      ReduceOp,                                                      \
      kOnnxDomain,                                                   \
      sinceVersion,                                                  \
      float,                                                         \
      kJsExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()) \
          .InputMemoryType(OrtMemTypeCPU, 1),                        \
      ReduceOp<float>);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMean, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 11, 11);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 12, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMax, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 11, 11);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 12, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceMin, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceProd, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceSum, 13);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceL1, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceL2, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceLogSum, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceSumSquare, 18);

REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 1, 10);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 11, 12);
REGISTER_REDUCE_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 13, 17);
REGISTER_REDUCE_ELEMENTWISE_KERNEL(ReduceLogSumExp, 18);

}  // namespace js
}  // namespace onnxruntime
