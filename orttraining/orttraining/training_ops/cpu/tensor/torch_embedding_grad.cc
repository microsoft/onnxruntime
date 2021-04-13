// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TORCH

#include "orttraining/training_ops/cpu/tensor/torch_embedding_grad.h"
#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    TorchEmbeddingGrad, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).ExternalOutputs(),
    TorchEmbeddingGrad);

Status TorchEmbeddingGrad::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue grad = *p_ctx_internal->GetInputMLValue(0);
  OrtValue indices = *p_ctx_internal->GetInputMLValue(1);
  int64_t num_weights = *p_ctx->Input<Tensor>(2)->Data<int64_t>();
  int64_t padding_idx = -1;
  if (p_ctx->InputCount() >= 4) {
    padding_idx = *p_ctx->Input<Tensor>(3)->Data<int64_t>();
  }

  bool scale_grad_by_freq = false;
  if (p_ctx->InputCount() >= 5) {
    scale_grad_by_freq = *p_ctx->Input<Tensor>(4)->Data<bool>();
  }

  at::Tensor torch_grad = dlpack::ToTorchTensor(grad);
  at::Tensor torch_indices = dlpack::ToTorchTensor(indices);
  auto torch_result =
      at::embedding_backward(torch_grad, torch_indices, num_weights, padding_idx, scale_grad_by_freq, false);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::FromTorchTensor(torch_result)));
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
