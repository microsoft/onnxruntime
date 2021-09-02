// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    1,
    10,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// axes is input instead of attribute
ONNX_CPU_OPERATOR_KERNEL(
    Squeeze,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

std::vector<int64_t> SqueezeBase::ComputeAxes(OpKernelContext* context, const std::vector<int64_t>& axes_attr) {
  std::vector<int64_t> axes;
  size_t num_inputs = context->InputCount();
  if (num_inputs == 2) {  //axes is an input
    const Tensor* axes_tensor = context->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->template Data<int64_t>();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_attr.begin(), axes_attr.end());
  }
  return axes;
}


}  // namespace onnxruntime
