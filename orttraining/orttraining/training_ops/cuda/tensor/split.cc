// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "split.h"
#include "core/providers/cuda/tensor/split_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_KERNEL_EX(SplitTraining,
                        kMSDomain,
                        1,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .InputMemoryType<OrtMemTypeCPUInput>(1)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        SplitTraining);

Status SplitTraining::PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
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

  std::vector<int64_t> split_sizes_values(split_sizes);
  split_sizes.clear();
  int64_t split_size_sum = std::accumulate(split_sizes_values.cbegin(), split_sizes_values.cend(), 0LL);

  if (split_sizes_values.empty()) {
    // equal split based on number of outputs
    if (split_dim_size % static_cast<size_t>(num_outputs) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input cannot be split evenly on selected axis. Input shape=", input_shape,
                             " Axis=", axis_, " NumOutputs=", num_outputs);
    }

    // populate split_sizes with the same size for each output
    split_sizes = std::vector<int64_t>(static_cast<size_t>(num_outputs), split_dim_size / num_outputs);
  } else {
    if (split_sizes_values.size() != static_cast<size_t>(num_outputs) || split_size_sum != split_dim_size)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot split using values in 'split' input. Axis=", axis_,
                             " Input shape=", input_shape,
                             " NumOutputs=", num_outputs,
                             " Num entries in 'split' (must equal number of outputs) was ", split_sizes_values.size(),
                             " Sum of sizes in 'split' (must equal size of selected axis) was ", split_size_sum);

    split_sizes = split_sizes_values;
  }

  return Status::OK();
}

Status SplitTraining::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != input_tensor);
  auto& input_shape = input_tensor->Shape();
  auto num_outputs = ctx->OutputCount();
  int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());
  int before_dims = 0;
  int block_size_including_axis_dim = 0;
  int block_size_inside_axis_dim = 0;

  //override the attribute value with the input value for split_split
  const Tensor* split_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(split_tensor != nullptr, "Split input is null");
  ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "An split tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
  const auto* data = split_tensor->template Data<int64_t>();
  std::vector<int64_t> split_sizes(data, data + nDims);

  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        block_size_including_axis_dim,
                                        block_size_inside_axis_dim,
                                        split_sizes));

  auto input_data = input_tensor->DataRaw();

  auto& input_dims = input_shape.GetDims();
  std::vector<int64_t> output_dimensions{input_dims};

  CudaAsyncBuffer<void*> output_ptr(this, num_outputs);
  gsl::span<void*> output_ptr_span = output_ptr.CpuSpan();
  std::vector<int64_t> axis_dimension_input_output_mapping(input_dims[axis]);
  int index = 0;
  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    auto split_size = gsl::narrow<int>(split_sizes[i]);
    output_dimensions[axis] = split_size;

    Tensor* output = ctx->Output(i, TensorShape{output_dimensions});
    auto output_data = output->MutableDataRaw();
    output_ptr_span[i] = output_data;
    for (int j = 0; j < split_size; ++j) {
      axis_dimension_input_output_mapping.at(index++) = i;
    }
  }

  if (input_tensor->Shape().Size() > 0) {
    output_ptr.CopyToGpu();

    CudaAsyncBuffer<int64_t> split_sizes_gpu(this, split_sizes);
    split_sizes_gpu.CopyToGpu();

    std::vector<int64_t> split_sizes_range(split_sizes);
    for (size_t i = 1; i < split_sizes_range.size(); ++i) {
      split_sizes_range[i] += split_sizes_range[i - 1];
    }

    CudaAsyncBuffer<int64_t> split_sizes_range_gpu(this, split_sizes_range);
    split_sizes_range_gpu.CopyToGpu();

    CudaAsyncBuffer<int64_t> axis_dimension_input_output_mapping_gpu(this, axis_dimension_input_output_mapping);
    axis_dimension_input_output_mapping_gpu.CopyToGpu();

    size_t element_size = input_tensor->DataType()->Size();
    ORT_RETURN_IF_ERROR(SplitImpl(element_size,
                                  block_size_including_axis_dim,
                                  block_size_inside_axis_dim,
                                  split_sizes_gpu.GpuPtr(),
                                  split_sizes_range_gpu.GpuPtr(),
                                  axis_dimension_input_output_mapping_gpu.GpuPtr(),
                                  num_outputs,
                                  input_data,
                                  output_ptr.GpuPtr(),
                                  input_shape.Size()));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
