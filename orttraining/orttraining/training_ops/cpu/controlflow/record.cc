// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "record.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

template <int input_start, int output_start>
std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.push_back(std::pair<int, int>(input_start + i, output_start + i));
  }
  return aliases;
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
  const Tensor* event_id_tensor = ctx->Input<Tensor>(0);
  const int64_t event_id = *event_id_tensor->template Data<int64_t>();

  ORT_RETURN_IF_NOT(event_id != -1, "-1 is reserved for skip wait, so cannot be used in RecordEvent");

  OrtEventPool::GetInstance().SignalEvent(event_id);

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
