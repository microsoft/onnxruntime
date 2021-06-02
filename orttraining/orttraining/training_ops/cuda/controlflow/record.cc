// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/controlflow/record.h"
#include "core/providers/cpu/tensor/utils.h"
// Include event mechanism shared by CPU and GPU implementations.
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "orttraining/training_ops/cpu/controlflow/record.h"
#include "core/profile/profile.h"
#include "core/profile/context.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    RecordEvent,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0) /* Keep EventIdentifier in CPU */
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .VariadicAlias(1, 0),  // outputs and inputs are mapped one to one, with input offset by 1
    RecordEvent);

Status RecordEvent::ComputeInternal(OpKernelContext* ctx) const {
#ifdef ENABLE_NVTX_PROFILE
  const Tensor* event_id_tensor = ctx->Input<Tensor>(0);
  const int64_t event_id = *(event_id_tensor->template Data<int64_t>());

  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
  profile::NvtxRangeCreator range(
      "Batch-" + tag + " Record-" + std::to_string(event_id), profile::Color::Magenta);
  range.Begin();
#endif

  // Reuse CPU helper to record event because event tensor is a CPU tensor.
  onnxruntime::contrib::record_event_in_tensor(*ctx->Input<Tensor>(0));

  for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
    // This iteration copies (i-1)-th input to i-th output.
    const Tensor* X = ctx->Input<Tensor>(i_out + 1);
    const TensorShape& data_shape = X->Shape();
    Tensor* Y = ctx->Output(i_out, data_shape);
    CopyTensor(*X, *Y);
  }

#ifdef ENABLE_NVTX_PROFILE
  range.End();
#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
