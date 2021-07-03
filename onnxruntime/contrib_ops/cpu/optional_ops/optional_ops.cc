// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optional_ops.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(Optional, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        Optional);

ONNX_OPERATOR_KERNEL_EX(OptionalHasElement, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
                        OptionalHasElement);

ONNX_OPERATOR_KERNEL_EX(OptionalGetElement, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        OptionalGetElement);

Status Optional::Compute(OpKernelContext* ctx) const {
  // TODO:
  return Status::OK();
}

Status OptionalHasElement::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);

  // Output is a scalar
  auto* output_tensor = ctx->Output(0, {});
  output_tensor->MutableData<bool>()[0] = input_ort_value->IsAllocated();

  return Status::OK();
}

Status OptionalGetElement::Compute(OpKernelContext* /* ctx */) const {
  // TODO:
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
