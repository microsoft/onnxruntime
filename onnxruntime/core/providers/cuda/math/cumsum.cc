// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cumsum.h"
#include "cumsum_impl.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum);

Status CumSum::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);                             // input tensor
  const auto rank = static_cast<int64_t>(input->Shape().NumDimensions());  // the rank of the input/output
  const Tensor* axis_tensor = ctx->Input<Tensor>(1);                       // axis input tensor

  if (axis_tensor->Shape().NumDimensions() > 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be 0D or 1D");

  int32_t axis = axis_tensor->template Data<int32_t>()[0];  // the axis on which the accumulation is going to done
  // validate input
  if (axis < -rank || axis >= rank)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis should be in the range [", -rank, ",", rank, ") but got: ", axis);

  if (axis < 0)
    axis = static_cast<int32_t>(rank) + axis;

  TensorShape output_shape(input->Shape());
  auto& output_tensor = *ctx->Output(0, output_shape);  // output tensor

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
