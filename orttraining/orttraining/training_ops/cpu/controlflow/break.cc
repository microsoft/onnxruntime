// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/break.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    BreakOp, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).ExternalOutputs(), BreakOp);

Status BreakOp::Compute(OpKernelContext* ctx) const {
  ORT_UNUSED_PARAMETER(ctx);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
