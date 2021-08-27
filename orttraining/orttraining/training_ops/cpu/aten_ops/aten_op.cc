// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_executor.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(ATenOp, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        ATenOp);

Status ATenOp::Compute(OpKernelContext* p_ctx) const {
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

  int64_t context_id = 0;
  int64_t* p_context_id = requires_grad_.empty() ? nullptr : &context_id;
  for (size_t i = 0; i < requires_grad_.size(); i++) {
    ORT_ENFORCE(dlpacks[static_cast<size_t>(requires_grad_[i])]);
  }

  auto result = aten_ops::ATenOperatorExecutor::Instance().ExecuteATenOperator(op_name_, overload_name_, dlpacks,
                                                                               requires_grad_, p_context_id);
  for (size_t i = 0; i < result.size(); i++) {
    ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
  }

  if (p_context_id) {
    Tensor* context_id_tensor = p_ctx->Output(p_ctx->OutputCount() - 1, {1});
    int64_t* p_context_id_data = context_id_tensor->MutableData<int64_t>();
    *p_context_id_data = context_id;
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(ATenOpGrad, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
                        ATenOpGrad);

Status ATenOpGrad::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  OrtValue ort_value = *p_ctx_internal->GetInputMLValue(0);
  DLManagedTensor* dlpack = dlpack::OrtValueToDlpack(ort_value);
  int64_t context_id = *p_ctx->Input<Tensor>(1)->Data<int64_t>();
  auto result = aten_ops::ATenOperatorExecutor::Instance().ExecuteATenOpBackward(dlpack, context_id);
  for (size_t i = 0; i < result.size(); i++) {
    ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
