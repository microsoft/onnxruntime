// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather.h"
#include "gather_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
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
            p.indices_tensor->template Data<int32_t>(),                         \
            div_strides.GpuPtr(),                                               \
            reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),     \
            reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data), \
            p.output_tensor->Shape().Size());                                   \
      }                                                                         \
      return Status::OK();                                                      \
    }                                                                           \
    if (utils::IsPrimitiveDataType<int64_t>(Tin_type)) {                        \
      if (p.output_tensor->Shape().Size() > 0) {                                \
        GatherImpl(                                                             \
            input_block_size,                                                   \
            indices_max,                                                        \
            p.indices_tensor->template Data<int64_t>(),                         \
            div_strides.GpuPtr(),                                               \
            reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),     \
            reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data), \
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

  // Put the output_block_size and block_size into div_strides
  // for divmod calling in _GatherKernel to calculate the input index
  CudaAsyncBuffer<fast_divmod> div_strides(this, 2);
  gsl::span<fast_divmod> div_strides_span = div_strides.CpuSpan();
  div_strides_span[0] = fast_divmod(gsl::narrow_cast<int>(output_block_size));
  div_strides_span[1] = fast_divmod(gsl::narrow_cast<int>(block_size));
  ORT_RETURN_IF_ERROR(div_strides.CopyToGpu());

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

}  // namespace cuda
}  // namespace onnxruntime
