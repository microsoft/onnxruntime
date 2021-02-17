// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_nd.h"
#include "core/providers/cuda/tensor/gather_nd_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

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
    cudaStream_t stream,
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
  const auto num_slices_per_batch = num_slices / num_batches;

  const TIndex* const indices_data = indices_tensor->Data<TIndex>();

  std::vector<int64_t> sizes_from_slice_dims(num_slice_dims);
  {
    auto running_product = slice_size;
    for (int64_t i = 0; i < num_slice_dims; ++i) {
      sizes_from_slice_dims[num_slice_dims - 1 - i] = running_product;
      running_product *= input_shape[batch_dims + num_slice_dims - 1 - i];
    }
  }

  auto sizes_from_slice_dims_buffer = GetScratchBuffer<int64_t>(sizes_from_slice_dims.size());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
      sizes_from_slice_dims_buffer.get(),
      sizes_from_slice_dims.data(),
      sizes_from_slice_dims.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice, stream));

  input_slice_offsets_buffer = GetScratchBuffer<int64_t>(num_slices);

  TArray<int64_t> input_dims(input_shape.GetDims());

  ComputeSliceOffsetsImpl(
      stream,
      batch_dims,
      input_dims,
      num_slices,
      num_slices_per_batch,
      input_batch_stride,
      num_slice_dims,
      sizes_from_slice_dims_buffer.get(),
      indices_data,
      input_slice_offsets_buffer.get());

  return Status::OK();
}

#define REGISTER_KERNEL_VERSIONED_TYPED_GATHER_ND(TIndex, startver, endver) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                  \
      GatherND,                                                             \
      kOnnxDomain,                                                          \
      startver,                                                             \
      endver,                                                               \
      TIndex,                                                               \
      kCudaExecutionProvider,                                               \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T",                                              \
                          std::vector<MLDataType>{                          \
                              DataTypeImpl::GetTensorType<float>(),         \
                              DataTypeImpl::GetTensorType<double>(),        \
                              DataTypeImpl::GetTensorType<MLFloat16>(),     \
                              DataTypeImpl::GetTensorType<int64_t>(),       \
                          })                                                \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIndex>()),   \
      GatherND<TIndex>);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define GATHER_ND_T_TENSOR_TYPES              \
  { DataTypeImpl::GetTensorType<float>(),     \
    DataTypeImpl::GetTensorType<double>(),    \
    DataTypeImpl::GetTensorType<MLFloat16>(), \
    DataTypeImpl::GetTensorType<BFloat16>(),  \
    DataTypeImpl::GetTensorType<int64_t>() }
#define GATHER_ND_T_DATA_TYPES float, MLFloat16, double, int64_t, BFloat16
#else
#define GATHER_ND_T_TENSOR_TYPES              \
  { DataTypeImpl::GetTensorType<float>(),     \
    DataTypeImpl::GetTensorType<double>(),    \
    DataTypeImpl::GetTensorType<MLFloat16>(), \
    DataTypeImpl::GetTensorType<int64_t>() }
#define GATHER_ND_T_DATA_TYPES float, MLFloat16, double, int64_t
#endif

#define REGISTER_KERNEL_TYPED_GATHER_ND(TIndex, ver)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                          \
      GatherND,                                                           \
      kOnnxDomain,                                                        \
      ver,                                                                \
      TIndex,                                                             \
      kCudaExecutionProvider,                                             \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T", GATHER_ND_T_TENSOR_TYPES)                  \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIndex>()), \
      GatherND<TIndex>);

// TODO: decprecate GatherND-1 after updating training models to opset-12
#ifdef ENABLE_TRAINING
REGISTER_KERNEL_TYPED_GATHER_ND(int64_t, 1)
#endif
REGISTER_KERNEL_TYPED_GATHER_ND(int64_t, 13)
REGISTER_KERNEL_VERSIONED_TYPED_GATHER_ND(int64_t, 12, 12)

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

  // Compute
  int64_t num_slices;
  int64_t slice_size;
  IAllocatorUniquePtr<int64_t> input_slice_offsets_buffer;
  ORT_RETURN_IF_ERROR(PrepareCompute<TIndex>(Stream(),
                                             batch_dims_, input_shape, indices_shape, indices_tensor,
                                             num_slices, slice_size, input_slice_offsets_buffer));

  const void* const kernel_input_data = input_tensor->DataRaw();
  void* const kernel_output_data = output_tensor->MutableDataRaw();
  utils::MLTypeCallDispatcher<GATHER_ND_T_DATA_TYPES> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<GatherNDComputeImpl>(
      Stream(), num_slices, slice_size, kernel_input_data, kernel_output_data, input_slice_offsets_buffer.get());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
