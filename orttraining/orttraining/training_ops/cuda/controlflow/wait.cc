// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/utils.h"
// Include RecordEvent's utility functions shared by CPU and GPU implementations.
#include "orttraining/training_ops/cpu/controlflow/common.h"
// Include event mechanism shared by CPU and GPU implementations.
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "wait.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    WaitEvent,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)   /* CPU variable */
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(onnxruntime::contrib::AliasRange<1, 0>(0, 1024)),
    WaitEvent);

Status WaitEvent::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* event_id_tensor = ctx->Input<Tensor>(0);
  const int64_t event_id = *event_id_tensor->template Data<int64_t>();

  // -1 is reserved to skip wait event
  if (event_id != -1) {
    // Wait the event to be recorded by a RecordEvent operator.
    onnxruntime::contrib::OrtEventPool::GetInstance().WaitEvent(event_id);

    // BUGBUG: seems this would cause hang when a event is being waited more than once
    // Destory the recorded event.
    onnxruntime::contrib::OrtEventPool::GetInstance().ResetEvent(event_id);
  }

  for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
    // This iteration copies (i-1)-th input to i-th output.
    const Tensor* X = ctx->Input<Tensor>(i_out + 1);
    const TensorShape& data_shape = X->Shape();
    Tensor* Y = ctx->Output(i_out, data_shape);
    CopyTensor(*X, *Y);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
