// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/gather_nd_grad.h"
#include "orttraining/training_ops/cuda/tensor/gather_nd_grad_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define ALL_IEEE_FLOAT_TENSOR_TYPES {DataTypeImpl::GetTensorType<float>(),      \
                                     DataTypeImpl::GetTensorType<double>(),     \
                                     DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                     DataTypeImpl::GetTensorType<BFloat16>()}
#define ALL_IEEE_FLOAT_DATA_TYPES float, MLFloat16, double, BFloat16
#else
#define ALL_IEEE_FLOAT_TENSOR_TYPES DataTypeImpl::AllIEEEFloatTensorTypes()
#define ALL_IEEE_FLOAT_DATA_TYPES float, MLFloat16, double
#endif

#define REGISTER_KERNEL_TYPED_GATHER_ND_GRAD(TIndex)                                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                            \
      GatherNDGrad,                                                                                                         \
      kMSDomain,                                                                                                            \
      1,                                                                                                                    \
      TIndex,                                                                                                               \
      kCudaExecutionProvider,                                                                                               \
      KernelDefBuilder().TypeConstraint("T", ALL_IEEE_FLOAT_TENSOR_TYPES)                                                   \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIndex>())                                                    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())                                                     \
          .InputMemoryType<OrtMemTypeCPUInput>(0),                                                                          \
      GatherNDGrad<TIndex>);

REGISTER_KERNEL_TYPED_GATHER_ND_GRAD(int64_t)

template <typename T>
struct GatherNDGradComputeImpl {
  void operator()(cudaStream_t stream,
                  const int64_t num_slices,
                  const int64_t slice_size,
                  const void* const kernel_input_data,
                  void* const kernel_output_data,
                  int64_t* const input_slice_offsets_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    GatherNDGradImpl<CudaT>(stream,
                            num_slices, kernel_input_data,
                            kernel_output_data, slice_size,
                            input_slice_offsets_data);
  }
};

template <typename TIndex>
Status GatherNDGrad<TIndex>::ComputeInternal(OpKernelContext* context) const {
  auto shape_tensor = context->Input<Tensor>(0);
  auto indices_tensor = context->Input<Tensor>(1);
  auto update_tensor = context->Input<Tensor>(2);
  ORT_RETURN_IF(shape_tensor == nullptr, "shape_tensor != nullptr");
  ORT_RETURN_IF(indices_tensor == nullptr, "indices_tensor != nullptr");
  ORT_RETURN_IF(update_tensor == nullptr, "update_tensor != nullptr");

  auto indices_shape = indices_tensor->Shape();
  auto update_shape = update_tensor->Shape();

  if (indices_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices tensor must has rank larger than 0");
  }

  auto last_indices_dimension = batch_dims_ + indices_shape[indices_shape.NumDimensions() - 1];

  // Output
  auto shape_data = shape_tensor->Data<int64_t>();
  auto input_shape = TensorShape(shape_data, shape_tensor->SizeInBytes() / sizeof(shape_tensor->DataType()));

  if (last_indices_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  ORT_RETURN_IF_ERROR(CheckBatchDimensionsMatch(
      static_cast<size_t>(batch_dims_), {input_shape, indices_shape, update_shape}));

  auto output_tensor = context->Output(0, input_shape);

  // TODO this memset can be expensive, a sparse tensor representation would help here
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output_tensor->MutableDataRaw(), 0, output_tensor->SizeInBytes(), Stream()));

  // Compute
  int64_t num_slices;
  int64_t slice_size;
  IAllocatorUniquePtr<int64_t> input_slice_offsets_buffer;
  ORT_RETURN_IF_ERROR(PrepareCompute<TIndex>(Stream(),
                                             batch_dims_, input_shape, indices_shape, indices_tensor,
                                             num_slices, slice_size, input_slice_offsets_buffer));

  const void* const kernel_input_data = update_tensor->DataRaw();
  void* const kernel_output_data = output_tensor->MutableDataRaw();
  utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(update_tensor->GetElementType());
  t_disp.Invoke<GatherNDGradComputeImpl>(
      Stream(), num_slices, slice_size, kernel_input_data, kernel_output_data, input_slice_offsets_buffer.get());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
