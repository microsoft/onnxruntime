// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/split.h"

#include "core/common/narrow.h"
#include <gsl/gsl>
#include "core/common/safeint.h"
#include "core/framework/copy.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0,
    element_type_lists::All);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0,
    int32_t, int64_t);
}  // namespace op_kernel_type_control

using EnabledSplitDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    2,
    10,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_1_13);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_1_13);

// Opset 13 starts to supports 'split' as optional input.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    13,
    17,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_1_13);

// TODO: support unequal split and num_outputs
ONNX_CPU_OPERATOR_KERNEL(
    Split,
    18,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_18);

Status SplitImpl::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  auto& input_shape = input.Shape();
  auto num_outputs = context->OutputCount();
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  std::vector<int64_t> split_sizes;

  const Tensor* split_tensor = context->Input<Tensor>(1);
  if (split_tensor != nullptr) {
    // override the attribute value with the input value for split
    ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "The split tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
    const auto* data = split_tensor->Data<int64_t>();
    split_sizes.assign(data, data + nDims);
  } else {
    split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  }

  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        after_dims_including_split_axis,
                                        after_dims_excluding_split,
                                        split_sizes));

  const auto input_strides = StridesForTensor(input);

  // copy dimensions so we can update the selected axis in place
  auto output_dimensions = input_shape.AsShapeVector();

  SafeInt<ptrdiff_t> input_offset = 0;

  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    auto split_size = narrow<int>(split_sizes[i]);
    output_dimensions[narrow<size_t>(axis)] = split_size;

    Tensor* output = context->Output(i, TensorShape{output_dimensions});
    const auto output_strides = StridesForTensor(*output);

    ORT_RETURN_IF_ERROR(DispatchStridedCopy<EnabledSplitDataTypes>(context->GetOperatorThreadPool(),
                                                                   *output, /* dst_offset */ 0, output_strides,
                                                                   output->Shape(),
                                                                   input, input_offset, input_strides));

    input_offset += SafeInt<ptrdiff_t>(split_size) * after_dims_excluding_split;  // offset by the data we used in this iteration
  }

  return Status::OK();
}

}  // namespace onnxruntime
