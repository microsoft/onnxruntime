// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/gather_nd.h"
#include "contrib_ops/cuda/gather_nd_impl.h"

namespace onnxruntime {
namespace cuda {

#define TYPED_FUNCTION_CALL_FWD(T)                                                                                    \
  if (T_type == DataTypeImpl::GetType<T>()) {                                                                         \
    GatherNDImpl<ToCudaType<T>::MappedType>(N, input_data, output_data, element_to_copy, element_offsets_data.get()); \
  }

#define TYPED_FUNCTION_CALL_BWD(T)                                                                                        \
  if (T_type == DataTypeImpl::GetType<T>()) {                                                                             \
    GatherNDGradImpl<ToCudaType<T>::MappedType>(N, input_data, output_data, element_to_copy, element_offsets_data.get()); \
  }
template <typename Tind>
Status GatherNDBase::CommonComputeKernel(
    const int64_t last_indice_dimension,
    const int64_t axis,
    const TensorShape& input_shape,
    const Tensor* input_tensor,
    Tensor* output_tensor,
    const TensorShape& indice_shape,
    const Tensor* indice_tensor,
    const bool fwd) const {
  auto indice_offset = indice_tensor->Data<Tind>();
  auto N = indice_shape.Size() / (last_indice_dimension - axis);  // Times to copy;
  auto input_data = input_tensor->DataRaw();
  auto output_data = output_tensor->MutableDataRaw();
  auto element_to_copy = input_shape.SizeFromDimension(last_indice_dimension);

  //Element_index_counts
  std::vector<int64_t> element_index_counts(last_indice_dimension + axis, 0LL);
  for (int64_t i = 0; i < last_indice_dimension; ++i) {
    element_index_counts[i] = input_shape.SizeFromDimension(i + 1);
  }

  auto last_dim_size = indice_shape.SizeFromDimension(indice_shape.NumDimensions() - 1);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = axis - 1; i >= 0; --i) {
    element_index_counts[last_indice_dimension + i] = indice_shape.SizeFromDimension(i + 1) / last_dim_size;
  }

  //Compute the element_offset
  std::vector<int64_t> element_offsets(N, 0LL);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < N; ++i) {
    int64_t reminder = i;
    for (int64_t j = 0; j < axis_; ++j) {
      int64_t idx = reminder / element_index_counts[last_indice_dimension + j];
      element_offsets[i] += idx * element_index_counts[j];
      reminder -= (idx * element_index_counts[last_indice_dimension + j]);
    }
    for (int64_t j = axis_; j < last_indice_dimension; ++j) {
      auto indice = *(indice_offset + i * (last_indice_dimension - axis_) + (j - axis_));
      element_offsets[i] += indice * element_index_counts[j];
    }
  }

  //copy element_counts to GPU;
  auto element_offsets_data = GetScratchBuffer<int64_t>(element_offsets.size());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(element_offsets_data.get(), element_offsets.data(),
                                       element_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  //Call cuda kernel
  if (fwd) {
    MLDataType T_type = input_tensor->DataType();
    TYPED_FUNCTION_CALL_FWD(float);
    TYPED_FUNCTION_CALL_FWD(MLFloat16);
    TYPED_FUNCTION_CALL_FWD(double);
  } else {
    MLDataType T_type = input_tensor->DataType();
    TYPED_FUNCTION_CALL_BWD(float);
    TYPED_FUNCTION_CALL_BWD(MLFloat16);
    TYPED_FUNCTION_CALL_BWD(double);
  }

  //Release the cuda memory
  return Status::OK();
}

#define REGISTER_KERNEL_TYPED_GATHER_ND(Tind)                                                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                            \
      GatherND,                                                                                                             \
      kOnnxDomain,                                                                                                          \
      1,                                                                                                                    \
      Tind,                                                                                                                 \
      kCudaExecutionProvider,                                                                                               \
      KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),                                     \
                                              DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>())                                                      \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                                                                          \
      GatherND<Tind>);

REGISTER_KERNEL_TYPED_GATHER_ND(int64_t)
REGISTER_KERNEL_TYPED_GATHER_ND(int32_t)

template <typename Tind>
Status GatherND<Tind>::ComputeInternal(OpKernelContext* context) const {
  auto input_tensor = context->Input<Tensor>(0);
  auto indice_tensor = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input_tensor != nullptr);
  ORT_RETURN_IF_NOT(indice_tensor != nullptr);

  auto input_shape = input_tensor->Shape();
  auto indice_shape = indice_tensor->Shape();

  if (indice_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices tensor must has rank larger than 0");
  }

  auto last_indice_dimension = axis_ + indice_shape[indice_shape.NumDimensions() - 1];
  if (last_indice_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  //Output shape
  std::vector<int64_t> shape(indice_shape.GetDims().begin(), indice_shape.GetDims().end() - 1);
  shape.insert(shape.end(), input_shape.GetDims().begin() + last_indice_dimension, input_shape.GetDims().end());

  auto output_tensor = context->Output(0, TensorShape(shape));

  //Compute
  auto status = CommonComputeKernel<Tind>(last_indice_dimension, axis_, input_shape, input_tensor, output_tensor, indice_shape, indice_tensor, true);

  return status;
}

#define REGISTER_KERNEL_TYPED_GATHER_ND_GRAD(Tind)                                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                            \
      GatherNDGrad,                                                                                                         \
      kOnnxDomain,                                                                                                          \
      1,                                                                                                                    \
      Tind,                                                                                                                 \
      kCudaExecutionProvider,                                                                                               \
      KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),                                     \
                                              DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>())                                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())                                                     \
          .InputMemoryType<OrtMemTypeCPUInput>(0)                                                                           \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                                                                          \
      GatherNDGrad<Tind>);

REGISTER_KERNEL_TYPED_GATHER_ND_GRAD(int64_t)
REGISTER_KERNEL_TYPED_GATHER_ND_GRAD(int32_t)

template <typename Tind>
Status GatherNDGrad<Tind>::ComputeInternal(OpKernelContext* context) const {
  auto shape_tensor = context->Input<Tensor>(0);
  auto indice_tensor = context->Input<Tensor>(1);
  auto update_tensor = context->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(shape_tensor != nullptr);
  ORT_RETURN_IF_NOT(indice_tensor != nullptr);
  ORT_RETURN_IF_NOT(update_tensor != nullptr);

  auto indice_shape = indice_tensor->Shape();
  auto update_shape = update_tensor->Shape();

  if (indice_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices tensor must has rank larger than 0");
  }

  auto last_indice_dimension = axis_ + indice_shape[indice_shape.NumDimensions() - 1];

  //Output
  auto shape_data = shape_tensor->Data<int64_t>();
  auto input_shape = TensorShape(shape_data, shape_tensor->Size() / sizeof(shape_tensor->DataType()));

  if (last_indice_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }
  auto output_tensor = context->Output(0, input_shape);
  CUDA_RETURN_IF_ERROR(cudaMemset(output_tensor->MutableDataRaw(), 0, output_tensor->Size()));

  auto status = CommonComputeKernel<Tind>(last_indice_dimension, axis_, input_shape, update_tensor, output_tensor, indice_shape, indice_tensor, false);
  return status;
}

}  // namespace cuda
}  // namespace onnxruntime
