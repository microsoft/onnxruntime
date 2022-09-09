// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/group.h"

namespace onnxruntime {
namespace contrib {

Status Group::Compute(OpKernelContext* context) const {
  Tensor& output = *context->Output(0, TensorShape({1}));
  bool* output_data = output.template MutableData<bool>();
  *output_data = true;

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Group,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Group);

Status PassThrough::Compute(OpKernelContext* context) const {
  for (int i = 0; i < context->InputCount(); ++i) {
    const auto* X = context->Input<Tensor>(i);
    ORT_ENFORCE(X != nullptr);
    Tensor* Y = context->Output(i, X->Shape());
    ORT_ENFORCE(X->DataRaw() == Y->DataRaw(), "PassThrough input and output are not sharing the same buffer.");
  }
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    PassThrough,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .VariadicAlias(0, 0),  // outputs and inputs are mapped one to one
    PassThrough);

}  // namespace contrib
}  // namespace onnxruntime
