// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/controlflow/pass_through.h"
#include "core/providers/cpu/tensor/utils.h"
#include "orttraining/training_ops/cpu/controlflow/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    PassThrough,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<MLFloat16>(),
                               DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<double>()})
        .Alias(onnxruntime::contrib::AliasRange<0, 0>(0, 1024)),
    PassThrough);

Status PassThrough::ComputeInternal(OpKernelContext* ctx) const {

  ORT_ENFORCE(ctx->InputCount() == ctx->OutputCount() + 1,
              "PassThrough's input tensor count must be the same as output tensor count.");

  for (int i = 0; i < ctx->OutputCount(); ++i) {
    const Tensor* X = ctx->Input<Tensor>(i + 1);
    const TensorShape& data_shape = X->Shape();
    Tensor* Y = ctx->Output(i, data_shape);
    CopyTensor(*X, *Y);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
