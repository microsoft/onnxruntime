// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/pass_through.h"
#include "core/providers/cpu/tensor/utils.h"
#include "orttraining/training_ops/cpu/controlflow/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    PassThrough,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
        .Alias(AliasRange<0, 0>(0, 1024)),
    PassThrough);

Status PassThrough::Compute(OpKernelContext* ctx) const {

  ORT_ENFORCE(ctx->InputCount() >= ctx->OutputCount(),
              "PassThrough's input count must be greater or equal to output count.");

  for (int i = 0; i < ctx->OutputCount(); ++i) {
    const Tensor* X = ctx->Input<Tensor>(i);
    const TensorShape& data_shape = X->Shape();
    Tensor* Y = ctx->Output(i, data_shape);
    if (!Y)
      continue;
    // This will do copy if not the same buffer.
    CopyCpuTensor(X, Y);
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
