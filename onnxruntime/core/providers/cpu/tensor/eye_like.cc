// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/eye_like.h"
#include "core/framework/tensorprotoutils.h"
#include "core/util/math_cpuonly.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    EyeLike,
    9,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{
                                          DataTypeImpl::GetTensorType<float>(),
                                          DataTypeImpl::GetTensorType<double>(),
                                          DataTypeImpl::GetTensorType<uint64_t>(),
                                          DataTypeImpl::GetTensorType<int64_t>(),
                                          DataTypeImpl::GetTensorType<int32_t>()
                                      })
                        .TypeConstraint("T2",
                                        std::vector<MLDataType>{
                                            DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<uint64_t>(),
                                            DataTypeImpl::GetTensorType<int64_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>()
                                        }),
    EyeLike);

Status EyeLike::Compute(OpKernelContext* context) const {
  const auto* T1 = context->Input<Tensor>(0);
  ORT_ENFORCE(T1 != nullptr);

  auto output_tensor_dtype = has_dtype_ ? static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype_) : T1->GetElementType();
  switch (output_tensor_dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return ComputeImpl<float>(context, T1);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return ComputeImpl<double>(context, T1);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return ComputeImpl<int32_t>(context, T1);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return ComputeImpl<uint64_t>(context, T1);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return ComputeImpl<int64_t>(context, T1);
    default:
      ORT_THROW("Unsupported 'dtype' value: ", output_tensor_dtype);
  }
}

template <typename T>
Status EyeLike::ComputeImpl(OpKernelContext* context, const Tensor* T1) const {
  const std::vector<int64_t>& input_dims = T1->Shape().GetDims();
  if (input_dims.size() != 2) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "EyeLike : Input tensor dimension is not 2");
  }

  // set output tensor shape same as input tensor and set all values to zero
  auto* T2 = context->Output(0, input_dims);
  auto output_mat = EigenMatrixMapRowMajor<T>(
      T2->template MutableData<T>(),
      input_dims[0],
      input_dims[1]);
  output_mat.setZero();
  
  if ((k_ >= 0 && k_ >= input_dims[1]) || (k_ < 0 && std::abs(k_) >= input_dims[0])) {
    return Status::OK();
  }
  output_mat.diagonal(k_).array() = static_cast<T>(1);

  return Status::OK();
}
}  // namespace onnxruntime
