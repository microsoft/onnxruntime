// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/generator/constant_of_shape.h"
#include "gsl/gsl"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    ConstantOfShape,
    9,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<MLFloat16>(),
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
    ConstantOfShape);

#define FETCH_VALUE_DATA(c_type)                                                                            \
  {                                                                                                         \
    c_type val;                                                                                             \
    auto unpack_status = UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1);                            \
    ORT_ENFORCE(unpack_status.IsOK(), "Value attribute unpacking failed:", unpack_status.ErrorMessage());   \
    SetValue(sizeof(c_type), reinterpret_cast<void*>(&val));                                                \
  }

void onnxruntime::ConstantOfShapeBase::SetValueFromTensorProto(const ONNX_NAMESPACE::TensorProto& t_proto) {
  using namespace utils;
  ORT_ENFORCE(utils::HasDataType(t_proto));
  ORT_ENFORCE(TensorProto::DataType_IsValid(t_proto.data_type()));
  const auto tensor_type = static_cast<TensorProto_DataType>(t_proto.data_type());
  const void* const raw_data = utils::HasRawData(t_proto) ? t_proto.raw_data().data() : nullptr;
  const size_t raw_data_len = utils::HasRawData(t_proto) ? t_proto.raw_data().size() : 0;
  switch (tensor_type) {
    case TensorProto::BOOL:
      FETCH_VALUE_DATA(bool);
      break;
    case TensorProto::FLOAT:
      FETCH_VALUE_DATA(float);
      break;
    case TensorProto::FLOAT16:
      FETCH_VALUE_DATA(MLFloat16);
      break;
    case TensorProto::DOUBLE:
      FETCH_VALUE_DATA(double);
      break;
    case TensorProto::INT8:
      FETCH_VALUE_DATA(int8_t);
      break;
    case TensorProto::INT16:
      FETCH_VALUE_DATA(int16_t);
      break;
    case TensorProto::INT32:
      FETCH_VALUE_DATA(int32_t);
      break;
    case TensorProto::INT64:
      FETCH_VALUE_DATA(int64_t);
      break;
    case TensorProto::UINT8:
      FETCH_VALUE_DATA(uint8_t);
      break;
    case TensorProto::UINT16:
      FETCH_VALUE_DATA(uint16_t);
      break;
    case TensorProto::UINT32:
      FETCH_VALUE_DATA(uint32_t);
      break;
    case TensorProto::UINT64:
      FETCH_VALUE_DATA(uint64_t);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", tensor_type);
      break;
  }
}

#undef FETCH_VALUE_DATA


template <class T>
inline void FilloutOutput(T value, void* output_data, size_t size) {
  auto out = gsl::make_span(reinterpret_cast<T*>(output_data), size);
  std::fill(out.begin(), out.end(), value);
}

ConstantOfShapeBase::ConstantOfShapeBase(const OpKernelInfo& info){
  TensorProto t_proto;
  if (info.GetAttr<TensorProto>("value", &t_proto).IsOK()) {
    ORT_ENFORCE(t_proto.dims_size() == 1, "Must have a single dimension");
    ORT_ENFORCE(t_proto.dims()[0] == 1, "Must have a single dimension of 1");
    SetValueFromTensorProto(t_proto);
  } else {
    float f_value = 0.f;
    SetValue(sizeof(float), reinterpret_cast<void*>(&f_value));
  }
}

Status ConstantOfShapeBase::PrepareCompute(OpKernelContext* ctx, Tensor** output_tensor) const {
  const auto shape_tensor = ctx->Input<Tensor>(0);
  const auto& input_shape = shape_tensor->Shape();

  // If empty the output is a scalar with empty shape
  // TensorShape::Size() will still return 1 and we will output
  // one value
  std::vector<int64_t> output_dims;
  ORT_ENFORCE(input_shape.NumDimensions() > 0, "Must have a valid input shape.");

  const auto span = gsl::make_span(shape_tensor->Data<int64_t>(), input_shape.Size());
  output_dims.insert(output_dims.end(), span.cbegin(), span.cend());

  TensorShape output_shape(output_dims);
  (*output_tensor) = ctx->Output(0, output_shape);

  return Status::OK();
}

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {

  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));

  auto output_data = output_tensor->MutableDataRaw();
  const void* value_ptr = GetValuePtr();
  const auto size = output_tensor->Shape().Size();
  const auto element_size = output_tensor->DataType()->Size();
  switch (element_size) {
    case sizeof(int8_t):
      FilloutOutput(*(reinterpret_cast<const int8_t*>(value_ptr)), output_data, size);
      break;
    case sizeof(int16_t):
      FilloutOutput(*(reinterpret_cast<const int16_t*>(value_ptr)), output_data, size);
      break;
    case sizeof(int32_t):
      FilloutOutput(*(reinterpret_cast<const int32_t*>(value_ptr)), output_data, size);
      break;
    case sizeof(int64_t):
      FilloutOutput(*(reinterpret_cast<const int64_t*>(value_ptr)), output_data, size);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype with sizeof=: ", element_size);
      break;
  }

  return Status::OK();
}
}  // namespace onnxruntime
