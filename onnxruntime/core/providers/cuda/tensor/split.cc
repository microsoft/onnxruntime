// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/split.h"

#include "core/providers/cuda/tensor/split_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  2, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split_2_13);

// explicitly supports negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split_2_13);

// explicitly supports 'split' as optional input
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  13, 17,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split_2_13);

ONNX_OPERATOR_KERNEL_EX(Split,
                        kOnnxDomain,
                        18,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Split_18);

Status SplitKernel::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor);
  auto& input_shape = input_tensor->Shape();
  auto num_outputs = ctx->OutputCount();
  int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());
  int before_dims = 0;
  int block_size_including_axis_dim = 0;
  int block_size_inside_axis_dim = 0;
  std::vector<int64_t> split_sizes(num_outputs);

  const Tensor* split_tensor = ctx->Input<Tensor>(1);
  if (split_tensor) {
    ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "An split tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
    const int64_t* data = split_tensor->Data<int64_t>();
    split_sizes.assign(data, data + nDims);
  } else {
    split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  }

  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        block_size_including_axis_dim,
                                        block_size_inside_axis_dim,
                                        split_sizes));

  auto input_data = input_tensor->DataRaw();

  auto input_dims = input_shape.GetDims();
  auto output_dimensions{input_shape.AsShapeVector()};

  CudaAsyncBuffer<void*> output_ptr(this, num_outputs);
  gsl::span<void*> output_ptr_span = output_ptr.CpuSpan();
  TensorShapeVector axis_dimension_input_output_mapping(input_dims[axis]);
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

  if (input_tensor->Shape().Size() <= 0) return Status::OK();

  size_t element_size = input_tensor->DataType()->Size();
  if (std::all_of(split_sizes.begin(), split_sizes.end(), [&](int64_t size) { return size == split_sizes[0]; })) {
    if (num_outputs <= 32) {
      TArray<void*, 32> output_ptr_array(num_outputs);
      for (int i = 0; i < num_outputs; ++i) output_ptr_array[i] = output_ptr_span[i];
      ORT_RETURN_IF_ERROR(SplitSameSplitDimImpl(Stream(ctx), element_size, block_size_including_axis_dim,
                                                block_size_inside_axis_dim, split_sizes[0], num_outputs, input_data,
                                                output_ptr_array, static_cast<size_t>(input_shape.Size())));
    } else {
      ORT_RETURN_IF_ERROR(output_ptr.CopyToGpu(ctx->GetComputeStream()));
      ORT_RETURN_IF_ERROR(SplitSameSplitDimImpl(Stream(ctx), element_size, block_size_including_axis_dim,
                                                block_size_inside_axis_dim, split_sizes[0], num_outputs, input_data,
                                                output_ptr.GpuPtr(), static_cast<size_t>(input_shape.Size())));
    }
  } else {
    ORT_RETURN_IF_ERROR(output_ptr.CopyToGpu(ctx->GetComputeStream()));
    CudaAsyncBuffer<int64_t> split_sizes_gpu(this, split_sizes);
    ORT_RETURN_IF_ERROR(split_sizes_gpu.CopyToGpu(ctx->GetComputeStream()));
    std::vector<int64_t> split_sizes_range(split_sizes);
    for (size_t i = 1; i < split_sizes_range.size(); ++i) {
      split_sizes_range[i] += split_sizes_range[i - 1];
    }
    CudaAsyncBuffer<int64_t> split_sizes_range_gpu(this, split_sizes_range);
    ORT_RETURN_IF_ERROR(split_sizes_range_gpu.CopyToGpu(ctx->GetComputeStream()));
    CudaAsyncBuffer<int64_t> axis_dimension_input_output_mapping_gpu(this, axis_dimension_input_output_mapping);
    ORT_RETURN_IF_ERROR(axis_dimension_input_output_mapping_gpu.CopyToGpu(ctx->GetComputeStream()));
    ORT_RETURN_IF_ERROR(SplitImpl(Stream(ctx), element_size, block_size_including_axis_dim, block_size_inside_axis_dim,
                                  split_sizes_gpu.GpuPtr(), split_sizes_range_gpu.GpuPtr(),
                                  axis_dimension_input_output_mapping_gpu.GpuPtr(), num_outputs, input_data,
                                  output_ptr.GpuPtr(), static_cast<size_t>(input_shape.Size())));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
