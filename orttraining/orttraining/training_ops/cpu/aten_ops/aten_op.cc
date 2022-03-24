// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_executor.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(ATen, kPytorchAtenDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        ATen);

Status ATen::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  std::vector<DLManagedTensor*> dlpacks;
  for (int i = 0; i < p_ctx_internal->InputCount(); i++) {
    const OrtValue* p_ort_value = p_ctx_internal->GetInputMLValue(i);
    if (!p_ort_value) {
      dlpacks.emplace_back(nullptr);
    } else {
      OrtValue ort_value = *p_ort_value;
      dlpacks.emplace_back(dlpack::OrtValueToDlpack(ort_value));
    }
  }

  auto result = aten_ops::ATenOperatorExecutor::Instance()(op_name_, overload_name_, dlpacks);
  for (size_t i = 0; i < result.size(); i++) {
    ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
  }

  return Status::OK();
}

bool IsATenOperatorExecutorInitialized() {
  return aten_ops::ATenOperatorExecutor::Instance().IsInitialized();
}

Status ExecuteReduceSumATen(OpKernelContext* p_ctx, const gsl::span<const int64_t>& axes, bool keepdims) {
  ORT_ENFORCE(aten_ops::ATenOperatorExecutor::Instance().IsInitialized() && !axes.empty());
  std::vector<DLManagedTensor*> dlpacks;
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue ort_value = *p_ctx_internal->GetInputMLValue(0);
  dlpacks.emplace_back(dlpack::OrtValueToDlpack(ort_value));
  OrtValue axes_tensor;
  OrtValue keepdims_tensor;
  TensorShapeVector axes_tensor_shape(1, static_cast<int64_t>(axes.size()));
  TensorShapeVector keepdims_tensor_shape(1, 1);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  OrtMemoryInfo info("Cpu", OrtDeviceAllocator);
  auto axes_tensor_obj = std::make_unique<Tensor>(DataTypeImpl::GetType<int64_t>(), axes_tensor_shape,
                              const_cast<void*>(reinterpret_cast<const void*>(&axes[0])), info);
  axes_tensor.Init(axes_tensor_obj.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  auto keepdims_tensor_obj = std::make_unique<Tensor>(DataTypeImpl::GetType<bool>(), keepdims_tensor_shape, reinterpret_cast<void*>(&keepdims), info);
  keepdims_tensor.Init(keepdims_tensor_obj.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  dlpacks.emplace_back(dlpack::OrtValueToDlpack(axes_tensor));
  dlpacks.emplace_back(dlpack::OrtValueToDlpack(keepdims_tensor));
  dlpacks.emplace_back(nullptr);
  auto result = aten_ops::ATenOperatorExecutor::Instance()("aten::sum", "dim_IntList", dlpacks);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, dlpack::DlpackToOrtValue(result[0])));
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
