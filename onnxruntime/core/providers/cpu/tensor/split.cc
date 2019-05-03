// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/split.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "gsl/gsl_util"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Split,
    2,
    KernelDefBuilder().TypeConstraint("T",
                                      std::vector<MLDataType>{
                                          DataTypeImpl::GetTensorType<float>(),
                                          DataTypeImpl::GetTensorType<double>(),
                                          DataTypeImpl::GetTensorType<int32_t>(),
                                      }),
    Split);

Status SplitBase::PrepareForCompute(const TensorShape& input_shape,
                                    const int num_outputs,
                                    int64_t& axis,
                                    int& before_dims,
                                    int& after_dims_including_split_axis,
                                    int& after_dims_excluding_split,
                                    std::vector<int64_t>& split_sizes) const {
  auto& input_dims = input_shape.GetDims();
  const int64_t num_dimensions = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
  axis = HandleNegativeAxis(axis_, num_dimensions);  // handle negative and enforce axis is valid
  const int64_t split_dim_size = input_dims[axis];

  before_dims = gsl::narrow<int>(input_shape.SizeToDimension(axis));
  after_dims_including_split_axis = gsl::narrow<int>(input_shape.SizeFromDimension(axis));
  after_dims_excluding_split = (axis + 1 == num_dimensions)
                                       ? 1  // we multiply by this value so must be 1 not 0
                                       : gsl::narrow<int>(input_shape.SizeFromDimension(axis + 1));

  if (split_sizes_.empty()) {
    // equal split based on number of outputs
    if (split_dim_size % num_outputs != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input cannot be split evenly on selected axis. Input shape=", input_shape,
                             " Axis=", axis_, " NumOutputs=", num_outputs);
    }

    // populate split_sizes with the same size for each output
    split_sizes = std::vector<int64_t>(num_outputs, split_dim_size / num_outputs);
  } else {
    if (split_sizes_.size() != num_outputs || split_size_sum_ != split_dim_size)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot split using values in 'split' attribute. Axis=", axis_,
                             " Input shape=", input_shape,
                             " NumOutputs=", num_outputs,
                             " Num entries in 'split' (must equal number of outputs) was ", split_sizes_.size(),
                             " Sum of sizes in 'split' (must equal size of selected axis) was ", split_size_sum_);

    split_sizes = split_sizes_;
  }

  return Status::OK();
}

Status Split::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  Status status;
  auto data_type = input.DataType();

  if (data_type == DataTypeImpl::GetType<float>())
    status = ComputeImpl<float>(*context, input);
  else if (data_type == DataTypeImpl::GetType<int32_t>())
    status = ComputeImpl<int32_t>(*context, input);
  else if (data_type == DataTypeImpl::GetType<double>()) {
    /* Need to update CopyMatrix to support double...
    status = ComputeImpl<double>(*context, input); */
    ORT_NOT_IMPLEMENTED("Split operator does not support double yet");
  } else
    ORT_THROW("Invalid data type for Split operator of ", data_type);

  return status;
}

template <typename T>
Status Split::ComputeImpl(OpKernelContext& context, const Tensor& input) const {
  auto& input_shape = input.Shape();
  auto num_outputs = context.OutputCount();
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  std::vector<int64_t> split_sizes;

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
          memcpy(dst, src, count * sizeof(T));
        });

    input_offset += split_size * after_dims_excluding_split;  // offset by the N data we used in this iteration
  }

  return Status::OK();
}

}  // namespace onnxruntime
