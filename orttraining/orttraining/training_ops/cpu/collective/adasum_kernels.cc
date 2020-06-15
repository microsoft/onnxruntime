// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/collective/adasum_kernels.h"

namespace onnxruntime {
namespace contrib {

AdasumAllReduce::AdasumAllReduce(const OpKernelInfo& info) : OpKernel(info) {
}

Status AdasumAllReduce::Compute(OpKernelContext* context) const {
  const Tensor& input_tensor = *context->Input<Tensor>(0);
  const void* input_data = input_tensor.DataRaw();
  const auto& input_shape = input_tensor.Shape();

  Tensor& output_tensor = *context->Output(0, input_shape);
  void* output_data = output_tensor.MutableDataRaw();

  if (output_data != input_data) {
    memcpy(output_data, input_data, input_tensor.SizeInBytes());
  }

  return Status::OK();
}

static std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.push_back(std::pair<int, int>(i, i));
  }
  return aliases;
}

ONNX_OPERATOR_KERNEL_EX(
    AdasumAllReduce,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(AliasRange(0, 1024))
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    AdasumAllReduce);

}  // namespace cuda
}  // namespace onnxruntime
