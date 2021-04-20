// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/external_functions/embedding_function.h"
#include "core/external_functions/external_function_registry.h"
#include "core/external_functions/attributes_json_parser.h"
#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace external_functions {

SPECIALIZED_EXTERNAL_FUNCTION_KERNEL_CREATOR(ATenEmbeddingFunction)
SPECIALIZED_EXTERNAL_FUNCTION_KERNEL_CREATOR(ATenEmbeddingBackwardFunction)

ATenEmbeddingFunction::ATenEmbeddingFunction(const OpKernelInfo& info, void* p_fn_raw) : OpKernel(info) {
  p_fn_ = reinterpret_cast<ATenEmbedding>(p_fn_raw);
  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  AttributesJsonParser parser(custom_attributes_json);
  padding_idx_ = parser.GetAttributeOrDefault<int>("padding_idx", -1);
  scale_grad_by_freq_ = parser.GetAttributeOrDefault<bool>("scale_grad_by_freq", false);
}

Status ATenEmbeddingFunction::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue weight = *p_ctx_internal->GetInputMLValue(0);
  OrtValue indices = *p_ctx_internal->GetInputMLValue(1);
  auto torch_result =
      p_fn_(dlpack::OrtValueToDlpack(weight), dlpack::OrtValueToDlpack(indices), padding_idx_, scale_grad_by_freq_);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::DlpackToOrtValue(torch_result)));
  return Status::OK();
}

ATenEmbeddingBackwardFunction::ATenEmbeddingBackwardFunction(const OpKernelInfo& info, void* p_fn_raw)
    : OpKernel(info) {
  p_fn_ = reinterpret_cast<ATenEmbeddingBackward>(p_fn_raw);
  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  AttributesJsonParser parser(custom_attributes_json);
  padding_idx_ = parser.GetAttributeOrDefault<int>("padding_idx", -1);
  scale_grad_by_freq_ = parser.GetAttributeOrDefault<bool>("scale_grad_by_freq", false);
}

Status ATenEmbeddingBackwardFunction::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue grad = *p_ctx_internal->GetInputMLValue(0);
  OrtValue weight = *p_ctx_internal->GetInputMLValue(1);
  OrtValue indices = *p_ctx_internal->GetInputMLValue(2);
  auto torch_result = p_fn_(dlpack::OrtValueToDlpack(grad), dlpack::OrtValueToDlpack(weight),
                            dlpack::OrtValueToDlpack(indices), padding_idx_, scale_grad_by_freq_);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::DlpackToOrtValue(torch_result)));
  return Status::OK();
}

}  // namespace external_functions
}  // namespace onnxruntime
