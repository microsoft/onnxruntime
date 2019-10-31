// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements.h"
#include "gather_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

#define TYPED_FUNCTION_CALL(T)                                                                                            \
  if (T_type == DataTypeImpl::GetType<T>()) {                                                                             \
    T* output_data = output_tensor->template MutableData<T>();                                                            \
    const T* input_data = input_tensor->template Data<T>();                                                               \
    if (Tin_type == DataTypeImpl::GetType<int32_t>()) {                                                                   \
      const int32_t* indices_data = indices_tensor->template Data<int32_t>();                                             \
      GatherElementsImpl(                                                                                                 \
          input_rank,                                                                                                     \
          reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),                                                 \
          input_size,                                                                                                     \
          input_dims[axis],                                                                                               \
          gpu_input_strides.GpuPtr(),                                                                                     \
          indices_data,                                                                                                   \
          indices_size,                                                                                                   \
          fdm_indices_strides.GpuPtr(),                                                                                   \
          axis,                                                                                                           \
          reinterpret_cast<ToCudaType<T>::MappedType*>(output_data));                                                     \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
    if (Tin_type == DataTypeImpl::GetType<int64_t>()) {                                                                   \
      const int64_t* indices_data = indices_tensor->template Data<int64_t>();                                             \
      GatherElementsImpl(                                                                                                 \
          input_rank,                                                                                                     \
          reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),                                                 \
          input_size,                                                                                                     \
          input_dims[axis],                                                                                               \
          gpu_input_strides.GpuPtr(),                                                                                     \
          indices_data,                                                                                                   \
          indices_size,                                                                                                   \
          fdm_indices_strides.GpuPtr(),                                                                                   \
          axis,                                                                                                           \
          reinterpret_cast<ToCudaType<T>::MappedType*>(output_data));                                                     \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherElements op: Type of 'indices' must be int32 or int64"); \
  }

Status GatherElements::ComputeInternal(OpKernelContext* context) const {
  // Process input data tensor
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();
  const auto& input_dims = input_shape.GetDims();
  const int64_t input_rank = static_cast<int64_t>(input_dims.size());
  const int64_t input_size = input_shape.Size();
  if (input_rank < 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Cannot operate on scalar input");

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, input_rank));

  // Process indices tensor
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto& indices_shape = indices_tensor->Shape();
  const auto& indices_dims = indices_shape.GetDims();
  const int64_t indices_rank = static_cast<int64_t>(indices_dims.size());
  const int64_t indices_size = indices_shape.Size();
  if (input_rank != indices_rank)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'");

  // Some shape checks for inidces and input tensors
  for (int64_t i = 0; i < indices_rank; ++i) {
    // for all axes except the axis of interest,
    // make sure that the corresponding 'indices' shape
    // value if within bounds of the corresponding 'data' shape
    if (i != axis) {
      if (indices_shape[i] < 0 || indices_shape[i] > input_shape[i])
        ORT_THROW(
            "GatherElements op: 'indices' shape should have values within bounds of 'data' shape. "
            "Invalid value in indices shape is: ",
            indices_shape[i]);
    }
  }

  // create output tensor
  auto* output_tensor = context->Output(0, TensorShape(indices_shape));

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  TensorPitches input_strides(input_dims);
  CudaAsyncBuffer<int64_t> gpu_input_strides(this, input_strides);

  CudaAsyncBuffer<fast_divmod> fdm_indices_strides(this, indices_rank);
  ORT_ENFORCE(CalculateFdmStrides(fdm_indices_strides.CpuSpan(), indices_dims));

  ORT_RETURN_IF_ERROR(gpu_input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_indices_strides.CopyToGpu());

  MLDataType T_type = input_tensor->DataType();
  MLDataType Tin_type = indices_tensor->DataType();

  // If one TYPED_FUNCTION_CALL finishes execution, the macro has a return statement in it,
  // and hence won't make subsequent checks
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

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "String type is not supported yet in GatherElements.");
}

}  // namespace cuda
}  // namespace onnxruntime
