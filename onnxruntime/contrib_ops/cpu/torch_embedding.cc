// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(USE_TORCH)

#include "contrib_ops/cpu/torch_embedding.h"
#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    TorchEmbedding, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).ExternalOutputs(), TorchEmbedding);

Status TorchEmbedding::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue weight = *p_ctx_internal->GetInputMLValue(0);
  OrtValue indices = *p_ctx_internal->GetInputMLValue(1);
  int64_t padding_idx = -1;
  if (p_ctx->InputCount() >= 3) {
    padding_idx = *p_ctx->Input<Tensor>(2)->Data<int64_t>();
  }

  bool scale_grad_by_freq = false;
  if (p_ctx->InputCount() >= 4) {
    scale_grad_by_freq = *p_ctx->Input<Tensor>(3)->Data<bool>();
  }

  at::Tensor torch_weight = dlpack::ToTorchTensor(weight);
  at::Tensor torch_indices = dlpack::ToTorchTensor(indices);
  auto torch_result = at::embedding(torch_weight, torch_indices, padding_idx, scale_grad_by_freq, false);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::FromTorchTensor(torch_result)));
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
