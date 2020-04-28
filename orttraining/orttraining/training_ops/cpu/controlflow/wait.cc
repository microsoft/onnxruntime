// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/wait.h"
#include "core/providers/cpu/tensor/utils.h"
#include "common.h"

namespace onnxruntime {
namespace contrib {

void wait_event_in_tensor(const Tensor& event_id_tensor) {
  const int64_t event_id = *event_id_tensor.template Data<int64_t>();
  // -1 is reserved to skip wait event
  if (event_id != -1) {
    // Wait the event to be recorded by a RecordEvent operator.
    OrtEventPool::GetInstance().WaitEvent(event_id);
    // BUGBUG: seems this would cause hang when a event is being waited more than once
    // Destory the recorded event.
    OrtEventPool::GetInstance().ResetEvent(event_id);
  }
}

ONNX_OPERATOR_KERNEL_EX(
    WaitEvent,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(AliasRange<1, 0>(0, 1024)),
    WaitEvent);

Status WaitEvent::Compute(OpKernelContext* ctx) const {
  wait_event_in_tensor(*ctx->Input<Tensor>(0));

  for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
    const Tensor* X = ctx->Input<Tensor>(i_out + 1);
    const TensorShape& data_shape = X->Shape();
    Tensor* Y = ctx->Output(i_out, data_shape);
    CopyCpuTensor(X, Y);
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
