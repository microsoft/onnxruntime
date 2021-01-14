// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "core/providers/cpu/tensor/utils.h"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(Yield, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Yield);

Status Yield::Compute(OpKernelContext* ctx) const {
  // FW output should be ready by this point, they are currently exposed as graph output
  // !!! Potential TODO here: If graph output approach doesn't work, need to place the Yield Input tensors into some shared location

  // single event for InferenceSession::RunInBackgroundAndWaitForYield() that FW graph is done
  const int64_t main_thread_event_id = 0;
  OrtEventPool::GetInstance().SignalEvent(main_thread_event_id);

  // wait for event from InferenceSession::ContinueRunInBackground() to continue the BW graph
  const int64_t background_thread_event_id = 1;
  OrtEventPool::GetInstance().ResetAndWaitEvent(background_thread_event_id);

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
