// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scatter_elements.h"
#include "scatter_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Scatter,
    kOnnxDomain,
    9,
    10,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    ScatterElements);

ONNX_OPERATOR_KERNEL_EX(
    ScatterElements,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    ScatterElements);

#define TYPED_FUNCTION_CALL(T)                                                \
  if (utils::IsPrimitiveDataType<T>(T_type)) {                                \
    T* output_data = output_tensor->template MutableData<T>();                \
    const T* input_data = data_tensor->template Data<T>();                    \
    const T* update_data = updates_tensor->template Data<T>();                \
    if (utils::IsPrimitiveDataType<int32_t>(Tin_type)) {                      \
      const int32_t* indices_data = indices_tensor->template Data<int32_t>(); \
      ScatterElementsImpl(                                                    \
          rank,                                                               \
          reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),     \
          input_data_size,                                                    \
          gpu_input_dims.GpuPtr(),                                            \
          gpu_input_strides.GpuPtr(),                                         \
          indices_data,                                                       \
          indices_size,                                                       \
          gpu_indices_dims.GpuPtr(),                                          \
          fdm_indices_strides.GpuPtr(),                                       \
          reinterpret_cast<const ToCudaType<T>::MappedType*>(update_data),    \
          axis,                                                               \
          reinterpret_cast<ToCudaType<T>::MappedType*>(output_data));         \
      return Status::OK();                                                    \
    }                                                                         \
    if (utils::IsPrimitiveDataType<int64_t>(Tin_type)) {                      \
      const int64_t* indices_data = indices_tensor->template Data<int64_t>(); \
      ScatterElementsImpl(                                                    \
          rank,                                                               \
          reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),     \
          input_data_size,                                                    \
          gpu_input_dims.GpuPtr(),                                            \
          gpu_input_strides.GpuPtr(),                                         \
          indices_data,                                                       \
          indices_size,                                                       \
          gpu_indices_dims.GpuPtr(),                                          \
          fdm_indices_strides.GpuPtr(),                                       \
          reinterpret_cast<const ToCudaType<T>::MappedType*>(update_data),    \
          axis,                                                               \
          reinterpret_cast<ToCudaType<T>::MappedType*>(output_data));         \
      return Status::OK();                                                    \
    }                                                                         \
  }

Status ScatterElements::ComputeInternal(OpKernelContext* context) const {
  const auto* data_tensor = context->Input<Tensor>(0);
  const auto& input_data_shape = data_tensor->Shape();
  const int64_t input_data_size = input_data_shape.Size();
  const int axis = static_cast<int>(HandleNegativeAxis(axis_, input_data_shape.NumDimensions()));

  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  if (data_tensor->DataType() != updates_tensor->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  const auto& indices_dims = indices_tensor->Shape().GetDims();
  const int64_t indices_size = indices_tensor->Shape().Size();
  const auto& updates_dims = updates_tensor->Shape().GetDims();
  if (indices_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indices and updates must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs updates dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", updates_dims[i]);
    }
  }

  // According to the spec the rank of ind/upd shall be the same as input(data)
  // and we also want to make sure that the dimensions of the of the ind/upd do not
  // exceed that of the input
  const auto& input_dims = input_data_shape.GetDims();
  if (input_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices must have the same rank as Input. Indices rank=",
                           indices_dims.size(), ". Input rank=", input_dims.size());
  }

  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than input dim=", input_dims[i]);
    }
  }

  int rank = (int)input_dims.size();
  auto* output_tensor = context->Output(0, input_data_shape);

  CudaAsyncBuffer<int64_t> gpu_input_dims(this, input_dims);
  TensorPitches input_strides(input_dims);
  CudaAsyncBuffer<int64_t> gpu_input_strides(this, input_strides);

  CudaAsyncBuffer<int64_t> gpu_indices_dims(this, indices_dims);
  CudaAsyncBuffer<fast_divmod> fdm_indices_strides(this, rank);
  ORT_ENFORCE(CalculateFdmStrides(fdm_indices_strides.CpuSpan(), indices_dims));

  ORT_RETURN_IF_ERROR(gpu_input_dims.CopyToGpu());
  ORT_RETURN_IF_ERROR(gpu_input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(gpu_indices_dims.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_indices_strides.CopyToGpu());

  MLDataType Tin_type = indices_tensor->DataType();
  MLDataType T_type = data_tensor->DataType();

  TYPED_FUNCTION_CALL(float)
  TYPED_FUNCTION_CALL(MLFloat16)
  TYPED_FUNCTION_CALL(int16_t)
  TYPED_FUNCTION_CALL(int8_t)
  TYPED_FUNCTION_CALL(int32_t)
  TYPED_FUNCTION_CALL(int64_t)
  TYPED_FUNCTION_CALL(uint8_t)
  TYPED_FUNCTION_CALL(uint16_t)
  TYPED_FUNCTION_CALL(uint32_t)
  TYPED_FUNCTION_CALL(uint64_t)
  TYPED_FUNCTION_CALL(double)
  TYPED_FUNCTION_CALL(bool)

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for T is not supported yet in ScatterElements.");
}

}  // namespace cuda
}  // namespace onnxruntime
