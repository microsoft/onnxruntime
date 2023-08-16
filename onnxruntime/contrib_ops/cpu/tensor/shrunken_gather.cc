// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.

#include "contrib_ops/cpu/tensor/shrunken_gather.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    ShrunkenGather,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind",
                        std::vector<MLDataType>{
                            DataTypeImpl::GetTensorType<int32_t>(),
                            DataTypeImpl::GetTensorType<int64_t>()}),
    ShrunkenGather);

void ShrunkenGatherCommon::CheckInput(const Tensor* input_tensor, const Tensor* indices_tensor, int64_t axis_in) const {
  const auto& input_shape = input_tensor->Shape();
  const auto& indices_shape = indices_tensor->Shape();

  ORT_ENFORCE(input_shape.NumDimensions() >= 1, "ShrunkenGather only support input with rank >= 1, got ",
              input_shape.NumDimensions(), "-D input");

  ORT_ENFORCE(indices_shape.NumDimensions() == 1, "ShrunkenGather only support 1D indices, got ",
              indices_shape.NumDimensions(), "-D indices");

  const auto input_rank = input_shape.NumDimensions();
  auto axis = HandleNegativeAxis(axis_in, narrow<int64_t>(input_rank));

  const int64_t N = indices_shape.Size();
  const int64_t indices_max = input_shape[gsl::narrow_cast<size_t>(axis)];
  ORT_ENFORCE(indices_max >= N, "ShrunkenGather indices elem count should <= input dim on axis: ", axis,
              ", got indices elem count:", N, " input dim: ", indices_max);
}

Status ShrunkenGather::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));
  ShrunkenGatherCommon::CheckInput(p.input_tensor, p.indices_tensor, p.axis);
  return Gather::Compute(context);
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
