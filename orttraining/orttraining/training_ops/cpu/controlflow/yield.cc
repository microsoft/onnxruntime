// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(Yield, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Yield);

Status Yield::Compute(OpKernelContext* ctx) const {
  int64_t event_id = std::chrono::system_clock::now().time_since_epoch().count();
  OrtEventPool::GetInstance().WaitEvent(event_id);

  // Do we need to return the event_id so main thread can use it to call SignalEvent()?

  // Get output grad from somewhere and prepare Op outputs.
  for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
    // TODO: need to get the tensor from somewhere.
    const Tensor* X = nullptr;
    const TensorShape& data_shape = X->Shape();
    Tensor* Y = ctx->Output(i_out, data_shape);
    // TODO: We may need CUDA kernel as GPU cannot use CopyCpuTensor.
    CopyCpuTensor(X, Y);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
