// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gistencode_op.h"

namespace onnxruntime {
namespace contrib {
ONNX_OPERATOR_KERNEL_EX(
    GistBinarizeEncoder,
	kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().Alias(0,0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    GistBinarizeEncoderOp);

Status GistBinarizeEncoderOp::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  Tensor* Y1 = context->Output(1, shape);
  auto X_type = X->DataType();
  auto* src = X->template Data<float>();
  auto* dst = Y1->template MutableData<bool>();
  for (int64_t i = 0; i < X->Shape().Size(); ++i) {
    dst[i] = src[i] > 0.0;
  }

  void* target = Y->MutableDataRaw(X_type);
  ORT_ENFORCE(target != nullptr);
  return Status::OK();
}
}
}
