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

  // Process indices tensor
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto& indices_shape = indices_tensor->Shape();
  const auto& indices_dims = indices_shape.GetDims();
  const int64_t indices_rank = static_cast<int64_t>(indices_dims.size());
  const int64_t indices_size = indices_shape.Size();

  // Handle negative axis if any
  const int64_t axis = static_cast<int64_t>(HandleNegativeAxis(axis_, input_rank));

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kenrel that hosts the shared checks)
  onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis);

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

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "String type is not supported yet for the GatherElements op");
}

}  // namespace cuda
}  // namespace onnxruntime
