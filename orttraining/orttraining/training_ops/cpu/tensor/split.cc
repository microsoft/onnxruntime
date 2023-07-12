// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/split.h"

#include "core/common/gsl.h"
#include "core/common/narrow.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    SplitTraining,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    SplitTraining);

Status PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                 int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                 std::vector<int64_t>& split_sizes) {
  auto input_dims = input_shape.GetDims();
  const auto num_dimensions = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
  const int64_t original_axis_value = axis;
  axis = HandleNegativeAxis(original_axis_value, num_dimensions);  // handle negative and enforce axis is valid
  const int64_t split_dim_size = input_dims[gsl::narrow_cast<size_t>(axis)];

  before_dims = narrow<int>(input_shape.SizeToDimension(gsl::narrow_cast<size_t>(axis)));
  after_dims_including_split_axis = narrow<int>(input_shape.SizeFromDimension(gsl::narrow_cast<size_t>(axis)));
  after_dims_excluding_split = (axis + 1 == num_dimensions)
                                   ? 1  // we multiply by this value so must be 1 not 0
                                   : narrow<int>(input_shape.SizeFromDimension(gsl::narrow_cast<size_t>(axis) + 1));

  std::vector<int64_t> split_sizes_values(split_sizes);
  split_sizes.clear();
  int64_t split_size_sum = std::accumulate(split_sizes_values.cbegin(), split_sizes_values.cend(), 0LL);

  if (split_sizes_values.empty()) {
    // equal split based on number of outputs
    if (split_dim_size % static_cast<size_t>(num_outputs) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input cannot be split evenly on selected axis. Input shape=", input_shape,
                             " Axis=", original_axis_value, " NumOutputs=", num_outputs);
    }

    // populate split_sizes with the same size for each output
    split_sizes = std::vector<int64_t>(static_cast<size_t>(num_outputs), split_dim_size / num_outputs);
  } else {
    if (split_sizes_values.size() != static_cast<size_t>(num_outputs) || split_size_sum != split_dim_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot split using values in 'split' input. Axis=", original_axis_value,
                             " Input shape=", input_shape,
                             " NumOutputs=", num_outputs,
                             " Num entries in 'split' (must equal number of outputs) was ", split_sizes_values.size(),
                             " Sum of sizes in 'split' (must equal size of selected axis) was ", split_size_sum);
    }
    split_sizes = split_sizes_values;
  }

  return Status::OK();
}

Status SplitTraining::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  Status status;

  if (input.IsDataType<float>())
    status = ComputeImpl<float>(*context, input);
  else if (input.IsDataType<int32_t>())
    status = ComputeImpl<int32_t>(*context, input);
  else if (input.IsDataType<int64_t>())
    status = ComputeImpl<int64_t>(*context, input);
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
Status SplitTraining::ComputeImpl(OpKernelContext& context, const Tensor& input) const {
  auto& input_shape = input.Shape();
  auto num_outputs = context.OutputCount();
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;

  // override the attribute value with the input value for split_split
  const Tensor* split_tensor = context.Input<Tensor>(1);
  ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "An split tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
  const auto* data = split_tensor->template Data<int64_t>();
  std::vector<int64_t> split_sizes(data, data + nDims);

  ORT_RETURN_IF_ERROR(PrepareForTrainingCompute(input_shape,
                                                num_outputs,
                                                axis,
                                                before_dims,
                                                after_dims_including_split_axis,
                                                after_dims_excluding_split,
                                                split_sizes));

  // copy dimensions so we can update the selected axis in place
  auto output_dimensions{input_shape.AsShapeVector()};

  int64_t input_offset = 0;
  const T* input_data = input.template Data<T>();

  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    auto split_size = narrow<int>(split_sizes[i]);
    output_dimensions[gsl::narrow_cast<size_t>(axis)] = split_size;

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

    input_offset += static_cast<int64_t>(split_size) * after_dims_excluding_split;  // offset by the N data we used in this iteration
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
