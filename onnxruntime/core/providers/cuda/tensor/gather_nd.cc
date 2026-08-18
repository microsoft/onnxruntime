// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_nd.h"
#include "core/providers/cuda/tensor/gather_nd_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

#include <algorithm>

namespace onnxruntime {
namespace cuda {

Status CheckBatchDimensionsMatch(
    size_t num_batch_dimensions,
    const std::vector<std::reference_wrapper<TensorShape>>& tensor_shapes) {
  for (size_t tensor_shape_idx = 0; tensor_shape_idx < tensor_shapes.size(); ++tensor_shape_idx) {
    const TensorShape& tensor_shape = tensor_shapes[tensor_shape_idx];
    ORT_RETURN_IF_NOT(
        num_batch_dimensions <= tensor_shape.NumDimensions(),
        "Number of batch dimensions exceeds tensor rank. ",
        "Batch dimension count: ", num_batch_dimensions,
        ", tensor rank: ", tensor_shape.NumDimensions(),
        ", tensor index: ", tensor_shape_idx);
  }

  if (tensor_shapes.empty()) return Status::OK();

  const TensorShape& first_tensor_shape = tensor_shapes.front();
  for (size_t batch_dimension_idx = 0; batch_dimension_idx < num_batch_dimensions; ++batch_dimension_idx) {
    for (size_t tensor_shape_idx = 1; tensor_shape_idx < tensor_shapes.size(); ++tensor_shape_idx) {
      const TensorShape& other_tensor_shape = tensor_shapes[tensor_shape_idx];
      ORT_RETURN_IF_NOT(
          first_tensor_shape[batch_dimension_idx] == other_tensor_shape[batch_dimension_idx],
          "Batch dimensions differ at index ", batch_dimension_idx, ": ",
          first_tensor_shape[batch_dimension_idx], " != ", other_tensor_shape[batch_dimension_idx],
          ", tensor indices: 0, ", tensor_shape_idx);
    }
  }

  return Status::OK();
}

template <typename TIndex>
Status GatherNDBase::PrepareCompute(
    void* alloc_stream,
    cudaStream_t cuda_stream,
    const int64_t batch_dims,
    const TensorShape& input_shape,
    const TensorShape& indices_shape,
    const Tensor* indices_tensor,
    int64_t& num_slices,
    int64_t& slice_size,
    IAllocatorUniquePtr<int64_t>& input_slice_offsets_buffer) const {
  const auto num_slice_dims = indices_shape[indices_shape.NumDimensions() - 1];
  num_slices = indices_shape.SizeToDimension(indices_shape.NumDimensions() - 1);
  slice_size = input_shape.SizeFromDimension(batch_dims + num_slice_dims);
  const auto num_batches = input_shape.SizeToDimension(batch_dims);
  const auto input_batch_stride = input_shape.SizeFromDimension(batch_dims);

  // Validate num_batches != 0 and num_slices divisibility to prevent division by zero
  ORT_RETURN_IF_NOT(num_batches != 0,
                    "Batch dimension cannot be zero");
  ORT_RETURN_IF_NOT(num_slices % num_batches == 0,
                    "Number of slices must be divisible by number of batches. ",
                    "num_slices = ", num_slices, ", num_batches = ", num_batches);
  const auto num_slices_per_batch = num_slices / num_batches;

  const TIndex* indices_data = indices_tensor->Data<TIndex>();
  IAllocatorUniquePtr<TIndex> indices_device_buffer;

  // Use on-device validation kernel to avoid full D2H copy for large indices tensors
  // This kernel records only the first invalid index into a 1-element device buffer,
  // then copies back only that value for error reporting.
  if (indices_tensor->Location().device.Type() == OrtDevice::CPU) {
    // For CPU-resident indices, fall back to host-side validation
    const size_t num_slices_size_t = static_cast<size_t>(num_slices);
    const size_t num_slice_dims_size_t = static_cast<size_t>(num_slice_dims);
    for (size_t slice_idx = 0; slice_idx < num_slices_size_t; ++slice_idx) {
      const size_t slice_base = slice_idx * num_slice_dims_size_t;
      for (size_t dim_idx = 0; dim_idx < num_slice_dims_size_t; ++dim_idx) {
        const int64_t index = static_cast<int64_t>(indices_data[slice_base + dim_idx]);
        const auto upper_limit = input_shape[batch_dims + static_cast<int64_t>(dim_idx)];
        const auto lower_limit = -upper_limit;
        ORT_RETURN_IF_NOT(index >= lower_limit && index < upper_limit,
                          "invalid index found, index = ", index);
      }
    }
  } else if (indices_tensor->Location().device.Type() == OrtDevice::GPU) {
    // Use on-device validation and copy back only the first invalid index for error reporting.
    TArray<int64_t> input_dims(input_shape.GetDims());
    auto validation_result_buffer = GetScratchBuffer<GatherNDValidationResult>(1, alloc_stream);
    const auto validation_result = ValidateIndicesAndReturnFirstInvalidIndex<TIndex>(
        cuda_stream,
        batch_dims,
        input_dims,
        num_slices,
        num_slice_dims,
        indices_data,
        validation_result_buffer.get());

    if (validation_result.position != -1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "invalid index found, index = ", validation_result.value);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported device type for indices tensor in CUDA GatherND");
  }

  if (indices_tensor->Location().device.Type() == OrtDevice::CPU) {
    indices_device_buffer = GetScratchBuffer<TIndex>(indices_tensor->Shape().Size(), alloc_stream);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(indices_device_buffer.get(),
                                         indices_data,
                                         indices_tensor->Shape().Size() * sizeof(TIndex),
                                         cudaMemcpyHostToDevice,
                                         cuda_stream));
    indices_data = indices_device_buffer.get();
  }

  // Pass strides by value so captured graphs do not retain a pointer to temporary host storage.
  TArray<int64_t> sizes_from_slice_dims(static_cast<int32_t>(num_slice_dims));
  {
    auto running_product = slice_size;
    for (int64_t i = 0; i < num_slice_dims; ++i) {
      sizes_from_slice_dims[static_cast<size_t>(num_slice_dims - 1 - i)] = running_product;
      running_product *= input_shape[batch_dims + num_slice_dims - 1 - i];
    }
  }

  input_slice_offsets_buffer = GetScratchBuffer<int64_t>(num_slices, alloc_stream);

  TArray<int64_t> input_dims(input_shape.GetDims());

  ComputeSliceOffsetsImpl(
      cuda_stream,
      batch_dims,
      input_dims,
      num_slices,
      num_slices_per_batch,
      input_batch_stride,
      num_slice_dims,
      sizes_from_slice_dims,
      indices_data,
      input_slice_offsets_buffer.get());

  return Status::OK();
}

#define REGISTER_KERNEL_VERSIONED_TYPED_GATHER_ND(TIndex, startver, endver)  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                   \
      GatherND,                                                              \
      kOnnxDomain,                                                           \
      startver,                                                              \
      endver,                                                                \
      TIndex,                                                                \
      kCudaExecutionProvider,                                                \
      (*KernelDefBuilder::Create())                                          \
          .TypeConstraint("T",                                               \
                          std::vector<MLDataType>{                           \
                              DataTypeImpl::GetTensorType<float>(),          \
                              DataTypeImpl::GetTensorType<double>(),         \
                              DataTypeImpl::GetTensorType<MLFloat16>(),      \
                              DataTypeImpl::GetTensorType<int64_t>(),        \
                              DataTypeImpl::GetTensorType<bool>(),           \
                          })                                                 \
          .TypeConstraint("indices", DataTypeImpl::GetTensorType<TIndex>()), \
      GatherND<TIndex>);

#define REGISTER_KERNEL_TYPED_GATHER_ND(TIndex, ver)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                               \
      GatherND, kOnnxDomain, ver, TIndex, kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                                                            \
          .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16, double, int64_t, BFloat16, bool>()) \
          .TypeConstraint("indices", DataTypeImpl::GetTensorType<TIndex>()),                                   \
      GatherND<TIndex>);

REGISTER_KERNEL_TYPED_GATHER_ND(int64_t, 13)
REGISTER_KERNEL_VERSIONED_TYPED_GATHER_ND(int64_t, 12, 12)
REGISTER_KERNEL_VERSIONED_TYPED_GATHER_ND(int64_t, 11, 11)

template <typename T>
struct GatherNDComputeImpl {
  void operator()(cudaStream_t stream,
                  const int64_t num_slices,
                  const int64_t slice_size,
                  const void* const kernel_input_data,
                  void* const kernel_output_data,
                  int64_t* const input_slice_offsets_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    GatherNDImpl<CudaT>(stream,
                        num_slices, kernel_input_data,
                        kernel_output_data, slice_size,
                        input_slice_offsets_data);
  }
};

template <typename TIndex>
Status GatherND<TIndex>::ComputeInternal(OpKernelContext* context) const {
  auto input_tensor = context->Input<Tensor>(0);
  auto indices_tensor = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input_tensor != nullptr, "input_tensor == nullptr");
  ORT_RETURN_IF_NOT(indices_tensor != nullptr, "indices_tensor == nullptr");

  auto input_shape = input_tensor->Shape();
  auto indices_shape = indices_tensor->Shape();

  if (indices_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices tensor must has rank larger than 0");
  }

  auto last_indices_dimension = batch_dims_ + indices_shape[indices_shape.NumDimensions() - 1];
  if (last_indices_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  ORT_RETURN_IF_ERROR(CheckBatchDimensionsMatch(
      static_cast<size_t>(batch_dims_), {input_shape, indices_shape}));

  // Output shape
  std::vector<int64_t> shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end() - 1);
  shape.insert(shape.end(), input_shape.GetDims().begin() + last_indices_dimension, input_shape.GetDims().end());

  auto output_tensor = context->Output(0, TensorShape(shape));

  // Bail out early in case the output is going to be empty
  if (output_tensor->Shape().Size() == 0) {
    return Status::OK();
  }

  // Compute
  int64_t num_slices;
  int64_t slice_size;
  IAllocatorUniquePtr<int64_t> input_slice_offsets_buffer;
  ORT_RETURN_IF_ERROR(PrepareCompute<TIndex>(GetComputeStream(context), Stream(context),
                                             batch_dims_, input_shape, indices_shape, indices_tensor,
                                             num_slices, slice_size, input_slice_offsets_buffer));

  const void* const kernel_input_data = input_tensor->DataRaw();
  void* const kernel_output_data = output_tensor->MutableDataRaw();
  utils::MLTypeCallDispatcher<float, MLFloat16, double, int64_t, BFloat16, bool> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<GatherNDComputeImpl>(Stream(context), num_slices, slice_size, kernel_input_data, kernel_output_data,
                                     input_slice_offsets_buffer.get());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
