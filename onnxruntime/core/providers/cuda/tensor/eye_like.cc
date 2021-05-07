// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "eye_like.h"
#include "eye_like_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    EyeLike,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>(), DataTypeImpl::GetTensorType<uint64_t>(), DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<int32_t>()}).TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>(), DataTypeImpl::GetTensorType<uint64_t>(), DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<int32_t>()}),
    EyeLike);

#define TYPED_FUNCTION_CALL(T)                                                              \
  EyeLikeImpl<typename ToCudaType<T>::MappedType>(                                          \
      Stream(),                                                                             \
      offset,                                                                               \
      dim1 + 1,                                                                             \
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(T2->template MutableData<T>()), \
      diag_count);                                                                          \
  break;

Status EyeLike::ComputeInternal(OpKernelContext* context) const {
  const auto* T1 = context->Input<Tensor>(0);
  ORT_ENFORCE(T1 != nullptr);

  const std::vector<int64_t>& input_dims = T1->Shape().GetDims();
  if (input_dims.size() != 2) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "EyeLike : Input tensor dimension is not 2");
  }

  // set output tensor shape same as input tensor and set all values to zero
  auto* T2 = context->Output(0, input_dims);
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(T2->MutableDataRaw(), 0, T2->SizeInBytes(), Stream()));
  auto dim0 = input_dims[0];
  auto dim1 = input_dims[1];

  if ((k_ >= 0 && k_ >= dim1) || (k_ < 0 && (std::abs(k_)) >= dim0)) {
    return Status::OK();
  }

  // Calculate the start offset and total number of elements in diagnal.
  size_t offset, diag_count;
  if (k_ >= 0) {
    offset = k_;
    diag_count = std::min(dim1 - k_, dim0);
  } else {
    offset = (-k_) * dim1;
    diag_count = std::min(dim0 + k_, dim1);
  }

  auto output_tensor_dtype = has_dtype_ ? static_cast<ONNX_NAMESPACE::TensorProto_DataType>(dtype_) : T1->GetElementType();
  switch (output_tensor_dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      TYPED_FUNCTION_CALL(float)
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      TYPED_FUNCTION_CALL(double)
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      TYPED_FUNCTION_CALL(int32_t)
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      TYPED_FUNCTION_CALL(uint64_t)
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      TYPED_FUNCTION_CALL(int64_t)
    default:
      ORT_THROW("Unsupported 'dtype' value: ", output_tensor_dtype);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
