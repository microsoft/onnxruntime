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
  OrtTasks::GetInstance().SetKernelOutputs(Status::OK(), TOKEN_YIELD_END_FORWARD, forward_outputs);

  // wait for data from SetExternalKernelOutputs() to continue executing the BW graph
  auto backward_inputs = OrtTasks::GetInstance().WaitForExternalKernelOutputs();
  bool terminate = backward_inputs.first;

  if (terminate) {
    ORT_THROW("Terminating backward run, since the terminate is set to true.");
  } else {
    ORT_ENFORCE(backward_inputs.second.size() == static_cast<size_t>(ctx->OutputCount()));
    for (int i = 0; i < ctx->OutputCount(); ++i) {
      if (std::find(full_shape_outputs_.begin(), full_shape_outputs_.end(), static_cast<int64_t>(i)) !=
          full_shape_outputs_.end()) {
        ORT_ENFORCE(ctx->Input<Tensor>(i)->Shape() == backward_inputs.second[i].Get<Tensor>().Shape());
      }
      ctx_internal->SetOutputMLValue(i, backward_inputs.second[i]);
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(Hole, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Hole);

// Hole executes when switching to Python for execution of a custom
// autograd function.  The implementation is very basic, just as a
// proof-of-concept for testing Megatron: we assume functions are
// single-input single-output, and we may need to do more to enforce
// ordering, and to manage inputs and outputs efficiently.

Status Hole::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  const OpKernelInfo& info = OpKernel::Info();
  int64_t external_fn_id;
  ORT_ENFORCE(info.GetAttr<int64_t>("external_fn", &external_fn_id).IsOK());
  int64_t is_backward;
  ORT_ENFORCE(info.GetAttr<int64_t>("is_backward", &is_backward).IsOK());

  // Pass data ORT->Python
  std::vector<OrtValue> forward_outputs;
  forward_outputs.reserve(ctx->InputCount());
  for (int i = 0; i < ctx->InputCount(); ++i) {
    forward_outputs.push_back(*ctx_internal->GetInputMLValue(i));
  }
  int32_t token_id = is_backward ? (TOKEN_HOLE_BACKWARD + external_fn_id) : (TOKEN_HOLE_FORWARD + external_fn_id);

  // return forward output and signal that FW graph is completed
  OrtTasks::GetInstance().SetKernelOutputs(Status::OK(), token_id, forward_outputs);
  // wait for data from SetHoleOutputs() to continue executing the ONNX graph.
  auto backward_inputs = OrtTasks::GetInstance().WaitForExternalKernelOutputs();
  bool terminate = backward_inputs.first;

  if (terminate) {
    ORT_THROW("Terminating backward run, since the terminate is set to true.");
  } else {
    ORT_ENFORCE(backward_inputs.second.size() == static_cast<size_t>(ctx->OutputCount()));
    for (int i = 0; i < ctx->OutputCount(); ++i) {
      if (std::find(full_shape_outputs_.begin(), full_shape_outputs_.end(), static_cast<int64_t>(i)) !=
          full_shape_outputs_.end()) {
        ORT_ENFORCE(ctx->Input<Tensor>(i)->Shape() == backward_inputs.second[i].Get<Tensor>().Shape());
      }
      ctx_internal->SetOutputMLValue(i, backward_inputs.second[i]);
    }
  }

  // // Signal that a portion of the graph is complete
  // const int64_t main_thread_event_id = 0;
  // OrtEventPool::GetInstance().SignalEvent(main_thread_event_id,
  //                                         is_backward ? (OrtEventPool::TOKEN_HOLE_BACKWARD + external_fn_id)
  //                                                     : (OrtEventPool::TOKEN_HOLE_FORWARD + external_fn_id));

  // // Wait for resumption from Python
  // const int64_t background_thread_event_id = 1;
  // onnxruntime::contrib::OrtEventPool::GetInstance().ResetAndWaitEvent(background_thread_event_id);

  // // Pass data Python->ORT
  // for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
  //   ctx_internal->SetOutputMLValue(i_out, onnxruntime::contrib::OrtMessageQueue::GetInstance().Pop());
  // }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
