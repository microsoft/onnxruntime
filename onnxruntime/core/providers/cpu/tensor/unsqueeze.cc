// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/unsqueeze.h"
#include "utils.h"
#include "core/providers/common.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    1,
    10,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Unsqueeze,
    11,
    12,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

// axes is input instead of attribute
ONNX_CPU_OPERATOR_KERNEL(
    Unsqueeze,
    13,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unsqueeze);

Status UnsqueezeBase::PrepareCompute(OpKernelContext* ctx, Prepare& p) const {
  const auto* X = ctx->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);
  auto& input_tensor = *X;

  std::vector<int64_t> axes;
  size_t num_inputs = ctx->InputCount();
  if (num_inputs == 2) {  //axes is an input
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 0 ||
                    axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a scalar or a 1-D tensor.");
    auto num_axes_elements = static_cast<size_t>(axes_tensor->Shape().Size());
    const auto* data = axes_tensor->template Data<int64_t>();
    axes.assign(data, data + num_axes_elements);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  // New dimension count is the current dimensions + the number of entries in axes
  // Initialize output_dims to 0 in each axis initially
  std::vector<int64_t> output_dims(axes.size() + input_tensor.Shape().NumDimensions(), 0);

  // Set all axes indices to 1 in output_dims and check for duplicates
  for (int64_t axis : axes) {
    // Valid axis range is [0, output_rank - 1]
    axis = HandleNegativeAxis(axis, output_dims.size());
    if (axis < 0 || axis >= static_cast<int64_t>(output_dims.size()))
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an out of range axis");
    if (output_dims[axis] != 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has a duplicate axis");
    output_dims[axis] = 1;
  }

  // Now fill in the zero entries with the existing shape
  {
    auto begin = input_tensor.Shape().GetDims().cbegin();
    for (auto& axisSize : output_dims) {
      if (axisSize == 0)
        axisSize = *begin++;
    }
    assert(begin == input_tensor.Shape().GetDims().cend());
  }

  TensorShape output_shape(output_dims);
  p.output_tensor = ctx->Output(0, output_shape);
  ORT_ENFORCE(nullptr != p.output_tensor);
  p.input_tensor = &input_tensor;
  return Status::OK();
}

Status Unsqueeze::Compute(OpKernelContext* ctx) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, p));
  CopyCpuTensor(p.input_tensor, p.output_tensor);
  return Status::OK();
}
}  // namespace onnxruntime
