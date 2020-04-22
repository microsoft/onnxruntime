// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/tensor/gather_impl.h"
#include "core/providers/hip/tensor/gather.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace hip {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1, 10,
    kHipExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

// explicit negative axis support
ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    11,
    kHipExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

#define TYPED_FUNCTION_CALL(T)                                                  \
  if (utils::IsPrimitiveDataType<T>(T_type)) {                                  \
    T* output_data = p.output_tensor->template MutableData<T>();                \
    const T* input_data = p.input_tensor->template Data<T>();                   \
    if (utils::IsPrimitiveDataType<int32_t>(Tin_type)) {                        \
      if (p.output_tensor->Shape().Size() > 0) {                                \
        GatherImpl(                                                             \
            input_block_size,                                                   \
            indices_max,                                                        \
            divmod_output_block_size,                                           \
            divmod_block_size,                                                  \
            p.indices_tensor->template Data<int32_t>(),                         \
            reinterpret_cast<const ToHipType<T>::MappedType*>(input_data),     \
            reinterpret_cast<typename ToHipType<T>::MappedType*>(output_data), \
            p.output_tensor->Shape().Size());                                   \
      }                                                                         \
      return Status::OK();                                                      \
    }                                                                           \
    if (utils::IsPrimitiveDataType<int64_t>(Tin_type)) {                        \
      if (p.output_tensor->Shape().Size() > 0) {                                \
        GatherImpl(                                                             \
            input_block_size,                                                   \
            indices_max,                                                        \
            divmod_output_block_size,                                           \
            divmod_block_size,                                                  \
            p.indices_tensor->template Data<int64_t>(),                         \
            reinterpret_cast<const ToHipType<T>::MappedType*>(input_data),     \
            reinterpret_cast<typename ToHipType<T>::MappedType*>(output_data), \
            p.output_tensor->Shape().Size());                                   \
      }                                                                         \
      return Status::OK();                                                      \
    }                                                                           \
  }

Status Gather::ComputeInternal(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_shape = p.input_tensor->Shape();

  const int64_t block_size = input_shape.SizeFromDimension(p.axis + 1);
  size_t N = p.indices_tensor->Shape().Size();
  const int64_t input_block_size = input_shape.SizeFromDimension(p.axis);
  const int64_t output_block_size = N * block_size;
  const int64_t indices_max = input_shape[p.axis];

  const fast_divmod divmod_output_block_size(gsl::narrow_cast<int>(output_block_size));
  const fast_divmod divmod_block_size(gsl::narrow_cast<int>(block_size));

  MLDataType T_type = p.input_tensor->DataType();
  MLDataType Tin_type = p.indices_tensor->DataType();

  TYPED_FUNCTION_CALL(int8_t)
  TYPED_FUNCTION_CALL(int16_t)
  TYPED_FUNCTION_CALL(int32_t)
  TYPED_FUNCTION_CALL(int64_t)
  TYPED_FUNCTION_CALL(uint8_t)
  TYPED_FUNCTION_CALL(uint16_t)
  TYPED_FUNCTION_CALL(uint32_t)
  TYPED_FUNCTION_CALL(uint64_t)
  TYPED_FUNCTION_CALL(MLFloat16)
  TYPED_FUNCTION_CALL(float)
  TYPED_FUNCTION_CALL(double)
  TYPED_FUNCTION_CALL(bool)

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

}  // namespace hip
}  // namespace onnxruntime
