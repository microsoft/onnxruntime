// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/identity_op.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    IdentityOp<true>);

ONNX_CPU_OPERATOR_KERNEL(
    Identity,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).Alias(0, 0),
    IdentityOp<false>);

void IdentityOpHelper::CopyInputToOutput(OpKernelContext* context, const Tensor& input, int output_index) {
  const TensorShape& shape = input.Shape();
  Tensor* Y = context->Output(output_index, shape);
  auto X_type = input.DataType();

  const void* source = input.DataRaw(X_type);
  void* target = Y->MutableDataRaw(X_type);

  //If source and target pointers are not equal, we need to copy the data.
  if (target != source) {
    if (X_type != DataTypeImpl::GetType<std::string>()) {
      memcpy(target, source, shape.Size() * X_type->Size());
    } else {
      // handle std::string
      const std::string* src = input.template Data<std::string>();
      std::string* dst = Y->template MutableData<std::string>();
      std::copy(src, src + shape.Size(), dst);
    }
  }
}

}  // namespace onnxruntime
