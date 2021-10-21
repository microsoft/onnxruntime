// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    YieldOp, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).ExternalOutputs(), YieldOp);

Status YieldOp::Compute(OpKernelContext* ctx) const {
  ORT_UNUSED_PARAMETER(ctx);
  return Status(common::ONNXRUNTIME, common::FAIL, "This operator should not be executed.");
}

}  // namespace contrib
}  // namespace onnxruntime
