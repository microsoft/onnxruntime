// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/torch/torch_script_op.h"

#include <dlpack/dlpack.h>
#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/training_ops/cpu/torch/torch_script_executor.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(TorchScript, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        TorchScript);

Status TorchScript::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());
  std::unique_ptr<DLManagedTensor*[]> dlpack_inputs = std::make_unique<DLManagedTensor*[]>(input_size);
  std::unique_ptr<DLManagedTensor*[]> dlpack_outputs = std::make_unique<DLManagedTensor*[]>(output_size);
  for (size_t i = 0; i < input_size; ++i) {
    const OrtValue* p_ort_value = p_ctx_internal->GetInputMLValue(static_cast<int>(i));
    ORT_ENFORCE(p_ort_value);
    OrtValue ort_value = *p_ort_value;
    dlpack_inputs[i] = dlpack::OrtValueToDlpack(ort_value);
  }

  torch::TorchScriptExecutor::Instance()(key_, script_, input_size, dlpack_inputs.get(), output_size,
                                         dlpack_outputs.get());
  for (size_t i = 0; i < output_size; ++i) {
    ORT_RETURN_IF_ERROR(
        p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(dlpack_outputs[i])));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
