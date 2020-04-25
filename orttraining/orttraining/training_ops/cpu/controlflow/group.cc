// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "group.h"

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

}  // namespace contrib
}  // namespace onnxruntime
