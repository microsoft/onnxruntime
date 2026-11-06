// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/aten_ops/aten_op.h"

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "contrib_ops/cpu/aten_ops/aten_op_executor.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(ATen, kPytorchAtenDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()), ATen);

Status ATen::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());
  std::unique_ptr<DLManagedTensor*[]> dlpack_inputs = std::make_unique<DLManagedTensor*[]>(input_size);
  std::unique_ptr<DLManagedTensor*[]> dlpack_outputs = std::make_unique<DLManagedTensor*[]>(output_size);
  for (size_t i = 0; i < input_size; ++i) {
    const OrtValue* p_ort_value = p_ctx_internal->GetInputMLValue(static_cast<int>(i));
    if (!p_ort_value) {
      dlpack_inputs[i] = nullptr;
    } else {
      OrtValue ort_value = *p_ort_value;
      dlpack_inputs[i] = dlpack::OrtValueToDlpack(ort_value);
    }
  }

  aten_ops::ATenOperatorExecutor::Instance()(op_name_, overload_name_, input_size, dlpack_inputs.get(), output_size,
                                             dlpack_outputs.get());
  for (size_t i = 0; i < output_size; ++i) {
    if (dlpack_outputs[i]) {
      ORT_RETURN_IF_ERROR(
          p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(dlpack_outputs[i])));
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
