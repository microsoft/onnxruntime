// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "orttraining/training_ops/cpu/controlflow/ort_tasks.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    YieldOp, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).ExternalOutputs(), YieldOp);

Status YieldOp::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  std::vector<OrtValue> forward_outputs;
  forward_outputs.reserve(ctx->InputCount());
  for (int i = 0; i < ctx->InputCount(); ++i) {
    forward_outputs.push_back(*ctx_internal->GetInputMLValue(i));
  }

  // return forward output and single that FW graph is completed
  OrtTasks::GetInstance().SetForwardOutputs(Status::OK(), forward_outputs);

  // wait for data from SetBackwardInputs() to continue executing the BW graph
  auto backward_inputs = OrtTasks::GetInstance().WaitForBackwardInputs();
  bool terminate = backward_inputs.first;

  if (terminate) {
    ORT_THROW("Terminating backward run, since the terminate is set to true.");
  } else {
    ORT_ENFORCE(backward_inputs.second.size() == static_cast<size_t>(ctx->OutputCount()));

    for (int i = 0, j = 0; i < ctx->InputCount(); ++i) {
      if (non_differentiable_outputs_[i]) {
        continue;
      }

      if (full_shape_outputs_[i]) {
        ORT_ENFORCE(ctx->Input<Tensor>(i)->Shape() == backward_inputs.second[j].Get<Tensor>().Shape());
      }
      ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(j, backward_inputs.second[j]));
      j++;
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
