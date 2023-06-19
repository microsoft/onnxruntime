//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License.

//
// Created by Jian Chen on 6/16/23.
//

#include "gather.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {
namespace js {
#define REGISTER_GATHER_VERSIONED_KERNEL(GatherOp, sinceVersion, endVersion)                             \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                                 \
      GatherOp,                                                                                                      \
      kOnnxDomain,                                                                                                   \
      sinceVersion, endVersion,                                                                                      \
      kJsExecutionProvider,                                                                                          \
      KernelDefBuilder()                                                                                 \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())                                              \
          .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}), \
      GatherOp);
#define REGISTER_GATHER_KERNEL(GatherOp, sinceVersion)                             \
  ONNX_OPERATOR_KERNEL_EX(                                                                                 \
      GatherOp,                                                                                                      \
      kOnnxDomain,                                                                                                   \
      sinceVersion,                                                                                      \
      kJsExecutionProvider,                                                                                          \
      KernelDefBuilder()                                                                                 \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())                                              \
          .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}), \
      GatherOp);

REGISTER_GATHER_VERSIONED_KERNEL(Gather, 1, 10);
REGISTER_GATHER_VERSIONED_KERNEL(Gather, 11, 12);
REGISTER_GATHER_KERNEL(Gather, 13);

REGISTER_GATHER_VERSIONED_KERNEL(GatherElements, 11, 12);
REGISTER_GATHER_KERNEL(GatherElements, 13);

REGISTER_GATHER_VERSIONED_KERNEL(GatherND, 11, 11);
REGISTER_GATHER_VERSIONED_KERNEL(GatherND, 12, 12);
REGISTER_GATHER_KERNEL(GatherND, 13);


}  // namespace js
}  // namespace onnxruntime
