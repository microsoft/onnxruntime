// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "constant.h"

#include "core/common/common.h"
#include "gsl/span"
#include "core/framework/tensorprotoutils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    ConstantLike,
    9,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<float>(),
                                  DataTypeImpl::GetTensorType<double>(),
                                  DataTypeImpl::GetTensorType<int8_t>(),
                                  DataTypeImpl::GetTensorType<int16_t>(),
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>(),
                                  DataTypeImpl::GetTensorType<uint8_t>(),
                                  DataTypeImpl::GetTensorType<uint16_t>(),
                                  DataTypeImpl::GetTensorType<uint32_t>(),
                                  DataTypeImpl::GetTensorType<uint64_t>(),
                                  DataTypeImpl::GetTensorType<bool>()})
        .TypeConstraint("T2", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<float>(),
                                  DataTypeImpl::GetTensorType<double>(),
                                  DataTypeImpl::GetTensorType<int8_t>(),
                                  DataTypeImpl::GetTensorType<int16_t>(),
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>(),
                                  DataTypeImpl::GetTensorType<uint8_t>(),
                                  DataTypeImpl::GetTensorType<uint16_t>(),
                                  DataTypeImpl::GetTensorType<uint32_t>(),
                                  DataTypeImpl::GetTensorType<uint64_t>(),
                                  DataTypeImpl::GetTensorType<bool>()}),
    ConstantLike);

template <typename T>
void GenerateData(Tensor& tensor, float value) {
  auto out = gsl::make_span(tensor.MutableData<T>(), tensor.Shape().Size());
  std::for_each(out.begin(), out.end(), [&value](T& v) { v = static_cast<T>(value); });
}

static Status GenerateConstantOutput(Tensor& Y, TensorProto::DataType dtype, float value) {
  switch (dtype) {
    case TensorProto::FLOAT:
      GenerateData<float>(Y, value);
      break;
    case TensorProto::DOUBLE:
      GenerateData<double>(Y, value);
      break;
    case TensorProto::INT8:
      GenerateData<int8_t>(Y, value);
      break;
    case TensorProto::INT16:
      GenerateData<int16_t>(Y, value);
      break;
    case TensorProto::INT32:
      GenerateData<int32_t>(Y, value);
      break;
    case TensorProto::INT64:
      GenerateData<int64_t>(Y, value);
      break;
    case TensorProto::UINT8:
      GenerateData<uint8_t>(Y, value);
      break;
    case TensorProto::UINT16:
      GenerateData<uint16_t>(Y, value);
      break;
    case TensorProto::UINT32:
      GenerateData<uint32_t>(Y, value);
      break;
    case TensorProto::UINT64:
      GenerateData<uint64_t>(Y, value);
      break;
    case TensorProto::BOOL:
      GenerateData<bool>(Y, value);
      break;
    case TensorProto::FLOAT16:
      ORT_NOT_IMPLEMENTED("FLOAT16 is not supported");
    default:
      ORT_THROW("Unsupported data type of ", dtype);
  }

  return Status::OK();
}

Status ConstantLike::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = nullptr;

  ONNX_NAMESPACE::TensorProto::DataType dtype = dtype_;
  if (nullptr != X) {
    Y = ctx->Output(0, X->Shape());
    if (dtype == ONNX_NAMESPACE::TensorProto::UNDEFINED) {
      dtype = utils::GetTensorProtoType(*X);
    }
  } else {
    ORT_ENFORCE(!shape_.empty(), "Neither Input tensor is not null nor shape attribute exists");
    Y = ctx->Output(0, TensorShape(shape_));
  }

  if (dtype == ONNX_NAMESPACE::TensorProto::UNDEFINED) {
    dtype = TensorProto_DataType_FLOAT;
  }

  return GenerateConstantOutput(*Y, dtype, value_);
}

}  // namespace onnxruntime
