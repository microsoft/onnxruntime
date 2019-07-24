// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "memcpy.h"
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

Memcpy::Memcpy(const OpKernelInfo& info)
    : OpKernel(info) {
  provider_ = info.GetExecutionProvider();
}

Status Memcpy::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());
  Status retval = provider_->CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());
  return retval;
}

}  // namespace onnxruntime
