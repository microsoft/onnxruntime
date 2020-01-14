// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "split.h"
#include "split_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  2, 10,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split);

// explicitly supports negative axis
ONNX_OPERATOR_KERNEL_EX(Split,
                        kOnnxDomain,
                        11,
                        kCudaExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Split);

Status Split::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != input_tensor);
  auto& input_shape = input_tensor->Shape();
  auto num_outputs = ctx->OutputCount();
  int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());
  int before_dims = 0;
  int block_size_including_axis_dim = 0;
  int block_size_inside_axis_dim = 0;
  std::vector<int64_t> split_sizes;

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

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
