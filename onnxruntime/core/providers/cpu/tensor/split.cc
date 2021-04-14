// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/split.h"

#include "gsl/gsl"

#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0,
    float, int32_t, int64_t, uint8_t, std::string);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0,
    int32_t, int64_t);
}  // namespace op_kernel_type_control

using SplitDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0);
using EnabledSplitDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    2,
    10,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<SplitDataTypes>(),
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<SplitDataTypes>(),
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split);

// Opset 13 starts to supports 'split' as optional input.
ONNX_CPU_OPERATOR_KERNEL(
    Split,
    13,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<SplitDataTypes>(),
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split);

Status SplitBase::PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                    int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                    std::vector<int64_t>& split_sizes) const {
  auto& input_dims = input_shape.GetDims();
  const auto num_dimensions = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
  axis = HandleNegativeAxis(axis_, num_dimensions);  // handle negative and enforce axis is valid
  const int64_t split_dim_size = input_dims[axis];

  before_dims = gsl::narrow<int>(input_shape.SizeToDimension(axis));
  after_dims_including_split_axis = gsl::narrow<int>(input_shape.SizeFromDimension(axis));
  after_dims_excluding_split = (axis + 1 == num_dimensions)
                                   ? 1  // we multiply by this value so must be 1 not 0
                                   : gsl::narrow<int>(input_shape.SizeFromDimension(axis + 1));

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

Status Split::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  Status status;

  // Note: The non-string implementations can probably be based on data type size.
  if (input.IsDataType<float>())
    status = ComputeImpl<float>(*context, input);
  else if (input.IsDataType<int32_t>())
    status = ComputeImpl<int32_t>(*context, input);
  else if (input.IsDataType<int64_t>())
    status = ComputeImpl<int64_t>(*context, input);
  else if (input.IsDataType<uint8_t>())
    status = ComputeImpl<uint8_t>(*context, input);
  else if (input.IsDataTypeString())
    status = ComputeImpl<std::string>(*context, input);
  else
    ORT_THROW("Split operator does not support ", input.DataType(), " yet");

  return status;
}

template <typename T>
inline void copy_data(const T* src, T* dst, size_t count) {
  memcpy(dst, src, count * sizeof(T));
}

template <>
inline void copy_data<std::string>(const std::string* src, std::string* dst, size_t count) {
  const std::string* end = src + count;
  std::copy(src, end, dst);
}

template <typename T>
Status Split::ComputeImpl(OpKernelContext& context, const Tensor& input) const {
  if (!utils::HasType<EnabledSplitDataTypes, T>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Data type is not supported in this build.");
  }

  auto& input_shape = input.Shape();
  auto num_outputs = context.OutputCount();
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  std::vector<int64_t> split_sizes;

  size_t num_inputs = context.InputCount();
  if (num_inputs == 2) {
    //override the attribute value with the input value for split_split
    const Tensor* split_tensor = context.Input<Tensor>(1);
    ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "An split tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
    const auto* data = split_tensor->template Data<int64_t>();
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

  // copy dimensions so we can update the selected axis in place
  auto& input_dims = input_shape.GetDims();
  std::vector<int64_t> output_dimensions{input_dims};

  int64_t input_offset = 0;
  const T* input_data = input.template Data<T>();

  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    auto split_size = gsl::narrow<int>(split_sizes[i]);
    output_dimensions[axis] = split_size;

    Tensor* output = context.Output(i, TensorShape{output_dimensions});
    T* output_data = output->template MutableData<T>();

    ::onnxruntime::math::CopyMatrix<T>(
        before_dims,                                       // M
        split_size * after_dims_excluding_split,           // N
        static_cast<const T*>(input_data + input_offset),  // A
        after_dims_including_split_axis,                   // lda
        static_cast<T*>(output_data),                      // B
        split_size * after_dims_excluding_split,           // ldb
        [](const T* src, T* dst, size_t count) {
          copy_data<T>(src, dst, count);
        });

    input_offset += split_size * after_dims_excluding_split;  // offset by the N data we used in this iteration
  }

  return Status::OK();
}

}  // namespace onnxruntime
