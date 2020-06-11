// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/record.h"
#include "core/providers/cpu/tensor/utils.h"
#include "common.h"

namespace onnxruntime {
namespace contrib {

void record_event_in_tensor(const Tensor& event_id_tensor) {
  const int64_t event_id = *event_id_tensor.template Data<int64_t>();
  // event_id -1 means no event should be recorded and this operator works
  // like an Identity operator.
  if (event_id != -1) {
    OrtEventPool::GetInstance().SignalEvent(event_id);
  }
}

ONNX_OPERATOR_KERNEL_EX(
    RecordEvent,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(AliasRange<1, 0>(0, 1024)),
    RecordEvent);

Status RecordEvent::Compute(OpKernelContext* ctx) const {
  // Pass event-id tensor to event-recording helper function.
  record_event_in_tensor(*ctx->Input<Tensor>(0));

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
