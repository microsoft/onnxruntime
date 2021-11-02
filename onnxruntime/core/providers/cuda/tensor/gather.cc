// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_impl.h"
#include "core/providers/cuda/tensor/gather.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1, 10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

// explicit negative axis support
ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

ONNX_OPERATOR_KERNEL_EX(
    GatherInternal,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // Set the output-1 to stay in CUDA_PINNED memory to avoid synchronous memcpy
        .OutputMemoryType(OrtMemTypeCPU, 1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Int32", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

using GatheredIndexIndex_t = int32_t;

Status Gather::ComputeInternal(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_shape = p.input_tensor->Shape();

  const int64_t block_size = input_shape.SizeFromDimension(p.axis + 1);
  size_t N = p.indices_tensor->Shape().Size();
  const int64_t input_block_size = input_shape.SizeFromDimension(p.axis);
  const int64_t output_block_size = N * block_size;
  const int64_t indices_max = input_shape[p.axis];
  const void* input_data = p.input_tensor->DataRaw();
  const void* indices_data = p.indices_tensor->DataRaw();
  void* output_data = p.output_tensor->MutableDataRaw();

  if (p.output_tensor->Shape().Size() == 0) {
    return Status::OK();
  }

  const fast_divmod divmod_output_block_size(gsl::narrow_cast<int>(output_block_size));
  const fast_divmod divmod_block_size(gsl::narrow_cast<int>(block_size));

  const size_t element_size = p.input_tensor->DataType()->Size();
  const size_t index_element_size = p.indices_tensor->DataType()->Size();

  // CUDA Kernel implementation supports element sizes of:
  // int8_t, int16_t, int32_t and int64_t which covers all supported
  // types since there is no computations necessary just data movement
  if (p.indices_tensor->IsDataType<int32_t>() ||
      p.indices_tensor->IsDataType<int64_t>()) {
    GatherImpl(
        Stream(),
        input_block_size,
        indices_max,
        divmod_output_block_size,
        divmod_block_size,
        indices_data,
        index_element_size,
        input_data,
        element_size,
        output_data,
        p.output_tensor->Shape().Size());

    auto* num_segments = context->Output(1, {1});
    int32_t* p_num_segments = num_segments->MutableData<int32_t>();

    const SafeInt<GatheredIndexIndex_t> num_gathered_indices{N};
    const int64_t& gather_dimension_size = indices_max;
    const int64_t& num_gathered_per_index = block_size;

    GatherGradPrepare<int64_t>(
      Stream(),
      CudaScratchBufferAllocator{*this},
      reinterpret_cast<const int64_t*>(indices_data),
      num_gathered_indices,
      gather_dimension_size,
      num_gathered_per_index,
      *p_num_segments);

    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

}  // namespace cuda
}  // namespace onnxruntime
