// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "orttraining/training_ops/cpu/controlflow/ort_tasks.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    YieldOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .VariadicAlias(0, 0),  // TODO: this is a hack to avoid allocating output buffer
    YieldOp);

Status YieldOp::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  std::vector<OrtValue> forward_outputs;
  for (int i = 0; i < ctx->InputCount(); ++i) {
    forward_outputs.push_back(*ctx_internal->GetInputMLValue(i));
  }

  LOGS(ctx->Logger(), WARNING) << "before SetForwardOutputs";

  // return forward output and single that FW graph is completed
  OrtTasks::GetInstance().SetForwardOutputs(Status::OK(), forward_outputs);

  LOGS(ctx->Logger(), WARNING) << "after SetForwardOutputs";

  // wait for data from SetBackwardInputs() to continue executing the BW graph
  std::vector<OrtValue> backward_inputs = OrtTasks::GetInstance().WaitForBackwardInputs();

  LOGS(ctx->Logger(), WARNING) << "after WaitForBackwardInputs";

  if (ctx_internal->GetTerminateFlag()) {
    LOGS(ctx->Logger(), WARNING) << "Resumed executing backward subgraph, terminate_flag is set to true.";
  } else {
    ORT_ENFORCE(backward_inputs.size() == static_cast<size_t>(ctx->OutputCount()));
    for (int i = 0; i < ctx->OutputCount(); ++i) {
      ctx_internal->SetOutputMLValue(i, backward_inputs[i]);
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
