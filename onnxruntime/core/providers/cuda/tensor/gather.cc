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
        .OutputMemoryType(OrtMemTypeCPU, 3)
        .OutputMemoryType(OrtMemTypeCPU, 4)
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

    if (context->OutputCount() > 1)
    {
        auto* num_segments = context->Output(1, {1});
        int32_t* p_num_segments = num_segments->MutableData<int32_t>();

        const SafeInt<GatheredIndexIndex_t> num_gathered_indices{N};
        const int64_t& gather_dimension_size = indices_max;
        const int64_t& num_gathered_per_index = block_size;

        IAllocatorUniquePtr<SegmentIndex_t> segment_offsets_out;
        SegmentIndex_t last_segment_partial_segment_offset_out;
        SegmentIndex_t last_segment_partial_segment_count_out;
        IAllocatorUniquePtr<SegmentIndex_t> per_segment_partial_segment_counts_out;
        IAllocatorUniquePtr<SegmentIndex_t> per_segment_partial_segment_offsets_out;

        auto gather_grad_prepare = [&]<typename T>() {
            IAllocatorUniquePtr<T> dX_indices_sorted_out, dY_indices_sorted_out;
            GatherGradPrepare<T>(
                Stream(),
                CudaScratchBufferAllocator{*this},
                reinterpret_cast<const T*>(indices_data),
                num_gathered_indices,
                gather_dimension_size,
                num_gathered_per_index,
                *p_num_segments, segment_offsets_out,
                last_segment_partial_segment_count_out,
                last_segment_partial_segment_offset_out,
                per_segment_partial_segment_counts_out,
                per_segment_partial_segment_offsets_out,
                dX_indices_sorted_out,
                dY_indices_sorted_out);

                auto* dX_indices_sorted = context->Output(7, p.indices_tensor->Shape());
                T* p_dX_indices_sorted = dX_indices_sorted->MutableData<T>();
                CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(p_dX_indices_sorted, dX_indices_sorted_out.get(),
                                                    dX_indices_sorted->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));

                auto* dY_indices_sorted = context->Output(8, p.indices_tensor->Shape());
                T* p_dY_indices_sorted = dY_indices_sorted->MutableData<T>();
                CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(p_dY_indices_sorted, dY_indices_sorted_out.get(),
                                                    dY_indices_sorted->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));

                return Status::OK();
        };

        if (p.indices_tensor->IsDataType<int32_t>())
        {
            ORT_RETURN_IF_ERROR(gather_grad_prepare.operator()<int32_t>());
        } else
        {
            ORT_RETURN_IF_ERROR(gather_grad_prepare.operator()<int64_t>());
        }

        auto* segment_offsets = context->Output(2, {*p_num_segments});
        int32_t* p_segment_offsets = segment_offsets->MutableData<int32_t>();
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(p_segment_offsets, segment_offsets_out.get(),
                                            segment_offsets->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));

        auto* last_segment_partial_segment_count = context->Output(3, {1});
        int32_t* p_last_segment_partial_segment_count = last_segment_partial_segment_count->MutableData<int32_t>();
        *p_last_segment_partial_segment_count = last_segment_partial_segment_count_out;

        auto* last_segment_partial_segment_offset = context->Output(4, {1});
        int32_t* p_last_segment_partial_segment_offset = last_segment_partial_segment_offset->MutableData<int32_t>();
        *p_last_segment_partial_segment_offset = last_segment_partial_segment_offset_out;

        auto* per_segment_partial_segment_counts = context->Output(5, {*p_num_segments});
        int32_t* p_per_segment_partial_segment_counts = per_segment_partial_segment_counts->MutableData<int32_t>();
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(p_per_segment_partial_segment_counts, per_segment_partial_segment_counts_out.get(),
                                            per_segment_partial_segment_counts->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));

        auto* per_segment_partial_segment_offsets = context->Output(6, {*p_num_segments});
        int32_t* p_per_segment_partial_segment_offsets = per_segment_partial_segment_offsets->MutableData<int32_t>();
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(p_per_segment_partial_segment_offsets, per_segment_partial_segment_offsets_out.get(),
                                            per_segment_partial_segment_offsets->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));
    }

    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

}  // namespace cuda
}  // namespace onnxruntime
