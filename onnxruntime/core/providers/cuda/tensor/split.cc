// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/split.h"
#include "core/providers/cuda/tensor/split_impl.h"
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
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split);

ONNX_OPERATOR_KERNEL_EX(Split,
                        kOnnxDomain,
                        13,
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
  size_t element_size = input_tensor->DataType()->Size();

  std::vector<int64_t> output_dimensions{input_dims};

  std::vector<void*> output_ptr(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    output_dimensions[axis] = gsl::narrow<int>(split_sizes[i]);

    Tensor* output = ctx->Output(i, TensorShape{output_dimensions});
    output_ptr[i] = output->MutableDataRaw();
  }

  if (input_shape.Size() == 0) return Status::OK();

  std::vector<int64_t> split_sizes_range(split_sizes);
  for (size_t i = 1; i < split_sizes_range.size(); ++i) {
    split_sizes_range[i] += split_sizes_range[i - 1];
  }

  if (num_outputs <= TArray<int64_t>::Capacity()) {
    TArray<void*> output_ptr_gpu(output_ptr);
    TArray<int64_t> split_sizes_gpu(split_sizes);
    TArray<int64_t> split_sizes_range_gpu(split_sizes_range);

    ORT_RETURN_IF_ERROR(SplitImpl(element_size,
                                  block_size_including_axis_dim,
                                  block_size_inside_axis_dim,
                                  split_sizes_gpu,
                                  split_sizes_range_gpu,
                                  num_outputs,
                                  input_data,
                                  output_ptr_gpu,
                                  input_shape.Size()));
  } else {
    CudaAsyncBuffer<void*> output_ptr_gpu(this, output_ptr);
    CudaAsyncBuffer<int64_t> split_sizes_gpu(this, split_sizes);
    CudaAsyncBuffer<int64_t> split_sizes_range_gpu(this, split_sizes_range);

    output_ptr_gpu.CopyToGpu();
    split_sizes_gpu.CopyToGpu();
    split_sizes_range_gpu.CopyToGpu();
    ORT_RETURN_IF_ERROR(SplitImpl(element_size,
                                  block_size_including_axis_dim,
                                  block_size_inside_axis_dim,
                                  split_sizes_gpu.ConstGpuPtr(),
                                  split_sizes_range_gpu.ConstGpuPtr(),
                                  num_outputs,
                                  input_data,
                                  output_ptr_gpu.GpuPtr(),
                                  input_shape.Size()));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
