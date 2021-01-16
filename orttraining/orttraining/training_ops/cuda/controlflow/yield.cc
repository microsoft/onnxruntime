// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/controlflow/yield.h"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "orttraining/training_ops/cpu/controlflow/message_queue.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Yield,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Yield);

Status Yield::ComputeInternal(OpKernelContext* ctx) const {
  // FW output should be ready by this point, they are currently exposed as graph output
  // !!! Potential TODO here: If graph output approach doesn't work, need to place the Yield Input tensors into some shared location

  // Do we need to synchronize here?
  //cudaStreamSynchronize(0);

  // single event for InferenceSession::RunInBackgroundAndWaitForYield() that FW graph is done
  const int64_t main_thread_event_id = 0;
  onnxruntime::contrib::OrtEventPool::GetInstance().SignalEvent(main_thread_event_id);

  // wait for event from InferenceSession::ContinueRunInBackground() to continue the BW graph
  const int64_t background_thread_event_id = 1;
  onnxruntime::contrib::OrtEventPool::GetInstance().ResetAndWaitEvent(background_thread_event_id);

  // Get output grad from somewhere and prepare Op outputs.
  const std::vector<OrtValue>& output_grads = onnxruntime::contrib::OrtMessageQueue::GetInstance().GetOutputGrads();
  for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
    OrtValue value = output_grads[i_out];
    const Tensor& X = value.Get<Tensor>();
    const TensorShape& data_shape = X.Shape();
    Tensor* Y = ctx->Output(i_out, data_shape);
    CopyTensor(X, *Y);
  }

  onnxruntime::contrib::OrtMessageQueue::GetInstance().ClearOutputGrads();
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
