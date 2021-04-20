// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/external_functions/external_function_op.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    ExternalFunctionOp, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    ExternalFunctionOpBase<false>);

ONNX_OPERATOR_KERNEL_EX(
    ExternalFunctionOpGrad, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    ExternalFunctionOpBase<true>);

template <bool is_backward>
Status ExternalFunctionOpBase<is_backward>::Compute(OpKernelContext* p_ctx) const {
  ORT_RETURN_IF_ERROR(external_function_->Compute(p_ctx));
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
