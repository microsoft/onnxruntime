// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "wait.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    WaitEvent,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>()),
    WaitEvent);

Status WaitEvent::Compute(OpKernelContext* ctx) const {
  const Tensor* event_id_tensor = ctx->Input<Tensor>(0);
  const int64_t event_id = *event_id_tensor->template Data<int64_t>();

  const Tensor* input_signal_tensor = ctx->Input<Tensor>(1);
  const bool input_signal = *input_signal_tensor->template Data<bool>();

  ORT_ENFORCE(input_signal, "Input signal must be true to trigger WaitEvent operator.");

  // Wait the event to be recorded by a RecordEvent operator.
  while (OrtEventPool::GetInstance().QueryEvent(event_id) != true) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  // Destory the recorded event.
  OrtEventPool::GetInstance().DeleteEvent(event_id);

  Tensor* output_signal_tensor = ctx->Output(0, {});
  *output_signal_tensor->template MutableData<bool>() = true;

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime