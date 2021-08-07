// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/concat_impl.h"
namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_KERNEL_EX(ConcatTraining,
                        kMSDomain,
                        1,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .OutputMemoryType(OrtMemTypeCPUInput, 1)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        ConcatTraining);

Status ConcatTraining::ComputeInternal(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, input_tensors, p));

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  std::vector<int64_t> concat_sizes(input_count);

  CudaAsyncBuffer<const void*> input_ptr(this, input_count);
  gsl::span<const void*> input_ptr_cpuspan = input_ptr.CpuSpan();
  std::vector<int64_t> axis_dimension_input_output_mapping(p.output_tensor->Shape()[p.axis]);
  int index = 0;
  for (int i = 0; i < input_count; ++i) {
    auto input = p.inputs[i];
    concat_sizes[i] = input.tensor->Shape()[p.axis];
    input_ptr_cpuspan[i] = input.tensor->DataRaw();
    for (int j = 0; j < input.tensor->Shape()[p.axis]; ++j) {
      axis_dimension_input_output_mapping.at(index++) = i;
    }
  }
  std::vector<int64_t> concat_sizes_range(concat_sizes);
  for (size_t i = 1; i < concat_sizes_range.size(); ++i) {
    concat_sizes_range[i] += concat_sizes_range[i - 1];
  }

  CudaAsyncBuffer<int64_t> concat_sizes_gpu(this, concat_sizes);
  CudaAsyncBuffer<int64_t> axis_dimension_input_output_mapping_gpu(this, axis_dimension_input_output_mapping);
  CudaAsyncBuffer<int64_t> concat_sizes_range_gpu(this, concat_sizes_range);
  concat_sizes_gpu.CopyToGpu();
  axis_dimension_input_output_mapping_gpu.CopyToGpu();
  concat_sizes_range_gpu.CopyToGpu();
  input_ptr.CopyToGpu();
  int block_size_inside_axis_dim = static_cast<int>(p.output_axis_pitch / p.output_tensor->Shape()[p.axis]);
  int block_size_including_axis_dim = static_cast<int>(p.output_axis_pitch);
  auto element_bytes = p.output_tensor->DataType()->Size();
  ORT_RETURN_IF_ERROR(ConcatImpl(Stream(),
                                 element_bytes,
                                 block_size_including_axis_dim,
                                 block_size_inside_axis_dim,
                                 concat_sizes_gpu.GpuPtr(),
                                 concat_sizes_range_gpu.GpuPtr(),
                                 axis_dimension_input_output_mapping_gpu.GpuPtr(),
                                 p.output_tensor->MutableDataRaw(),
                                 input_ptr.GpuPtr(),
                                 p.output_num_elements));

  // Create optional output tensor for 'per_input_length'
  Tensor* per_input_length_tensor = ctx->Output(1, {input_count});
  if (per_input_length_tensor) {
    std::copy(concat_sizes.begin(), concat_sizes.end(), per_input_length_tensor->template MutableData<int64_t>());
  }

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
