// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "core/providers/cpu/controlflow/event_pool.h"
#include "core/providers/cpu/controlflow/message_queue.h"
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

bool YieldOp::input_ = true;

Status YieldOp::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  if (input_) {
    for (int i_in = 0; i_in < ctx->InputCount(); ++i_in) {
      onnxruntime::contrib::OrtMessageQueue::GetInstance().Push(*ctx_internal->GetInputMLValue(i_in));
    }
    input_ = false;
  } else {
    input_ = true;

    // Reset background event before returning to main thread
    //const int64_t background_thread_event_id = 1;
    //onnxruntime::contrib::OrtEventPool::GetInstance().ResetEvent(background_thread_event_id);

    // single event for InferenceSession::RunInBackgroundAndWaitForYield() that FW graph is done
    //const int64_t main_thread_event_id = 0;
    //OrtEventPool::GetInstance().SignalEvent(main_thread_event_id);

    // wait for event from InferenceSession::ContinueRunInBackground() to continue the BW graph
    //OrtEventPool::GetInstance().WaitAndResetEvent(background_thread_event_id);

    //if (ctx_internal->GetTerminateFlag()) {
    //LOGS(ctx->Logger(), WARNING) << "Resumed executing backward subgraph, terminate_flag is set to true.";
    //} else {
    // Get output grad from somewhere and prepare Op outputs.
    for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
      ctx_internal->SetOutputMLValue(i_out, OrtMessageQueue::GetInstance().Pop());
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
