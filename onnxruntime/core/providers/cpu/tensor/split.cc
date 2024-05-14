// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/split.h"

#include "core/common/narrow.h"
#include "core/common/gsl.h"
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

Status SplitBase::PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                    int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                    std::vector<int64_t>& split_sizes) const {
  auto input_dims = input_shape.GetDims();
  const auto num_dimensions = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
  axis = HandleNegativeAxis(axis_, num_dimensions);  // handle negative and enforce axis is valid
  const int64_t split_dim_size = input_dims[onnxruntime::narrow<size_t>(axis)];

  before_dims = narrow<int>(input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis)));
  after_dims_including_split_axis = narrow<int>(input_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis)));
  after_dims_excluding_split = (axis + 1 == num_dimensions)
                                   ? 1  // we multiply by this value so must be 1 not 0
                                   : narrow<int>(input_shape.SizeFromDimension(SafeInt<size_t>(axis) + 1));

  if (num_outputs_ != -1) {
    if (num_outputs_ > split_dim_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid num_outputs value of ", num_outputs_,
                             ". Size of dimension being split is ", split_dim_size);
    }

    // populate split sizes based on num_outputs so existing code can be utilized
    int32_t size = narrow<int32_t>(std::ceil(float(split_dim_size) / num_outputs));
    int32_t remainder = split_dim_size % size;

    split_sizes = std::vector<int64_t>(num_outputs, size);
    if (remainder) {
      split_sizes.back() = remainder;
    }
  }

  if (split_sizes.empty()) {
    // equal split based on number of outputs
    if (split_dim_size % static_cast<size_t>(num_outputs) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input cannot be split evenly on selected axis. Input shape=", input_shape,
                             " Axis=", axis_, " NumOutputs=", num_outputs);
    }

    // populate split_sizes with the same size for each output
    split_sizes = std::vector<int64_t>(static_cast<size_t>(num_outputs), split_dim_size / num_outputs);
  } else {
    int64_t split_size_sum = split_size_sum_;
    if (split_size_sum == -1) {
      split_size_sum = std::accumulate(split_sizes.cbegin(), split_sizes.cend(), 0LL);
    }
    if (split_sizes.size() != static_cast<size_t>(num_outputs) || split_size_sum != split_dim_size)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot split using values in 'split' attribute. Axis=", axis_,
                             " Input shape=", input_shape,
                             " NumOutputs=", num_outputs,
                             " Num entries in 'split' (must equal number of outputs) was ", split_sizes.size(),
                             " Sum of sizes in 'split' (must equal size of selected axis) was ", split_size_sum);
  }
  return Status::OK();
}

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
