// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/concat.h"

#include "core/providers/cuda/tensor/concat_impl.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Concat,
                                  kOnnxDomain,
                                  4, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Concat);

// opset 11 explicitly support negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Concat,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Concat);

ONNX_OPERATOR_KERNEL_EX(Concat,
                        kOnnxDomain,
                        13,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Concat);

Status Concat::ComputeInternal(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, input_tensors, p));

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  std::vector<int64_t> concat_sizes;
  concat_sizes.reserve(input_count);

  CudaAsyncBuffer<const void*> input_ptr(this, input_count);
  gsl::span<const void*> input_ptr_cpuspan = input_ptr.CpuSpan();
  std::vector<int64_t> axis_dimension_input_output_mapping(p.output_tensor->Shape()[p.axis]);
  int index = 0;
  for (int i = 0; i < input_count; ++i) {
    const auto& input = p.inputs[i];
    concat_sizes.push_back(input.tensor->Shape()[p.axis]);
    input_ptr_cpuspan[i] = input.tensor->DataRaw();
    for (int j = 0; j < input.tensor->Shape()[p.axis]; ++j) {
      axis_dimension_input_output_mapping.at(index++) = i;
    }
  }

  auto element_bytes = p.output_tensor->DataType()->Size();
  int block_size_inside_axis_dim = static_cast<int>(p.output_axis_pitch / p.output_tensor->Shape()[p.axis]);
  int block_size_including_axis_dim = static_cast<int>(p.output_axis_pitch);
  if (std::all_of(concat_sizes.begin(), concat_sizes.end(), [&](int64_t size) { return size == concat_sizes[0]; })) {
    if (input_count <= 32) {
      TArray<const void*, 32> input_ptr_array(input_count);
      for (int i = 0; i < input_count; ++i) input_ptr_array[i] = input_ptr_cpuspan[i];
      ORT_RETURN_IF_ERROR(ConcatSameConcatDimImpl(
          Stream(ctx), element_bytes, block_size_including_axis_dim, block_size_inside_axis_dim, concat_sizes[0],
          p.output_tensor->MutableDataRaw(), input_ptr_array, static_cast<size_t>(p.output_num_elements)));
    } else {
      ORT_RETURN_IF_ERROR(input_ptr.CopyToGpu(ctx->GetComputeStream()));
      ORT_RETURN_IF_ERROR(ConcatSameConcatDimImpl(
          Stream(ctx), element_bytes, block_size_including_axis_dim, block_size_inside_axis_dim, concat_sizes[0],
          p.output_tensor->MutableDataRaw(), input_ptr.GpuPtr(), static_cast<size_t>(p.output_num_elements)));
    }
  } else {
    CudaAsyncBuffer<int64_t> concat_sizes_gpu(this, concat_sizes);
    CudaAsyncBuffer<int64_t> axis_dimension_input_output_mapping_gpu(this, axis_dimension_input_output_mapping);
    std::vector<int64_t> concat_sizes_range(concat_sizes);
    for (size_t i = 1; i < concat_sizes_range.size(); ++i) {
      concat_sizes_range[i] += concat_sizes_range[i - 1];
    }
    CudaAsyncBuffer<int64_t> concat_sizes_range_gpu(this, concat_sizes_range);
    ORT_RETURN_IF_ERROR(concat_sizes_gpu.CopyToGpu(ctx->GetComputeStream()));
    ORT_RETURN_IF_ERROR(axis_dimension_input_output_mapping_gpu.CopyToGpu(ctx->GetComputeStream()));
    ORT_RETURN_IF_ERROR(concat_sizes_range_gpu.CopyToGpu(ctx->GetComputeStream()));
    ORT_RETURN_IF_ERROR(input_ptr.CopyToGpu(ctx->GetComputeStream()));
    ORT_RETURN_IF_ERROR(ConcatImpl(Stream(ctx), element_bytes, block_size_including_axis_dim, block_size_inside_axis_dim,
                                   concat_sizes_gpu.GpuPtr(), concat_sizes_range_gpu.GpuPtr(),
                                   axis_dimension_input_output_mapping_gpu.GpuPtr(), p.output_tensor->MutableDataRaw(),
                                   input_ptr.GpuPtr(), static_cast<size_t>(p.output_num_elements)));
  }

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
