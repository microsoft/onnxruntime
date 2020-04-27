// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/controlflow/record.h"
#include "core/providers/cpu/tensor/utils.h"
// Include RecordEvent's utility functions shared by CPU and GPU implementations.
#include "orttraining/training_ops/cpu/controlflow/common.h"
// Include event mechanism shared by CPU and GPU implementations.
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "orttraining/training_ops/cpu/controlflow/record.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    RecordEvent,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)   /* Keep EventIdentifier in CPU */
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(onnxruntime::contrib::AliasRange<1, 0>(0, 1024)),
    RecordEvent);

Status RecordEvent::ComputeInternal(OpKernelContext* ctx) const {
  // Reuse CPU helper to record event because event tensor is a CPU tensor.
  onnxruntime::contrib::record_event_in_tensor(*ctx->Input<Tensor>(0));

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
