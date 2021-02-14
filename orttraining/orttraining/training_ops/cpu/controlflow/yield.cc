// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "orttraining/training_ops/cpu/controlflow/ort_tasks.h"
#include "orttraining/training_ops/cpu/controlflow/message_queue.h"
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

  int64_t run_id = std::hash<std::thread::id>()(std::this_thread::get_id());
  for (int i_in = 0; i_in < ctx->InputCount(); ++i_in) {
    OrtTasks::GetInstance().Push(run_id, *ctx_internal->GetInputMLValue(i_in));
  }

  // Reset background event before returning to main thread
  OrtTasks::GetInstance().PrepareBackgroundWait();

  // single event for InferenceSession::RunInBackgroundAndWaitForYield() that FW graph is done
  OrtTasks::GetInstance().WakeupForegroundThread();

  // wait for event from InferenceSession::ContinueRunInBackground() to continue the BW graph
  OrtTasks::GetInstance().WaitInBackgroundThread();

  if (ctx_internal->GetTerminateFlag()) {
    LOGS(ctx->Logger(), WARNING) << "Resumed executing backward subgraph, terminate_flag is set to true.";
  } else {
    // Get output grad from somewhere and prepare Op outputs.
    for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
      ctx_internal->SetOutputMLValue(i_out, OrtTasks::GetInstance().Pop(run_id));
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
