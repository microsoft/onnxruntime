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

#ifdef ENABLE_TRAINING
bool IsATenOperatorExecutorInitialized() { return aten_ops::ATenOperatorExecutor::Instance().IsInitialized(); }

Status ExecuteReduceSumATen(OpKernelContext* p_ctx, const gsl::span<const int64_t>& axes, bool keepdims) {
  ORT_ENFORCE(aten_ops::ATenOperatorExecutor::Instance().IsInitialized() && !axes.empty());
  size_t input_size = 4;
  std::unique_ptr<DLManagedTensor*[]> dlpack_inputs = std::make_unique<DLManagedTensor*[]>(input_size);
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue ort_value = *p_ctx_internal->GetInputMLValue(0);
  dlpack_inputs[0] = dlpack::OrtValueToDlpack(ort_value);
  OrtValue axes_tensor;
  OrtValue keepdims_tensor;
  TensorShapeVector axes_tensor_shape(1, static_cast<int64_t>(axes.size()));
  TensorShapeVector keepdims_tensor_shape(1, 1);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  OrtMemoryInfo info("Cpu", OrtDeviceAllocator);
  auto axes_tensor_obj = std::make_unique<Tensor>(DataTypeImpl::GetType<int64_t>(), axes_tensor_shape,
                                                  const_cast<void*>(reinterpret_cast<const void*>(&axes[0])), info);
  axes_tensor.Init(axes_tensor_obj.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  auto keepdims_tensor_obj = std::make_unique<Tensor>(DataTypeImpl::GetType<bool>(), keepdims_tensor_shape,
                                                      reinterpret_cast<void*>(&keepdims), info);
  keepdims_tensor.Init(keepdims_tensor_obj.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  dlpack_inputs[1] = dlpack::OrtValueToDlpack(axes_tensor);
  dlpack_inputs[2] = dlpack::OrtValueToDlpack(keepdims_tensor);
  dlpack_inputs[3] = nullptr;
  DLManagedTensor* dlpack_output = nullptr;
  aten_ops::ATenOperatorExecutor::Instance()("sum", "dim_IntList", input_size, dlpack_inputs.get(), 1, &dlpack_output);
  ORT_ENFORCE(dlpack_output);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::DlpackToOrtValue(dlpack_output)));
  return Status::OK();
}
#endif

}  // namespace contrib
}  // namespace onnxruntime
