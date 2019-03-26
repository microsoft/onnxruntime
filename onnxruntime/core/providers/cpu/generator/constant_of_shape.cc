// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/generator/constant_of_shape.h"
#include "gsl/span"

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

#define FETCH_VALUE_DATA(field, c_type)                                                                   \
  {                                                                                                       \
    c_type t;                                                                                             \
    auto unpack_status = UnpackTensor(t_proto, raw_data, raw_data_len, &t, 1);                            \
    ORT_ENFORCE(unpack_status.IsOK(), "Value attribute unpacking failed:", unpack_status.ErrorMessage()); \
    field = t;                                                                                            \
  }

void onnxruntime::ConstantOfShape::SetValue(const ONNX_NAMESPACE::TensorProto& t_proto) {
  using namespace utils;
  ORT_ENFORCE(t_proto.has_data_type());
  ORT_ENFORCE(TensorProto::DataType_IsValid(t_proto.data_type()));
  tensor_type_ = static_cast<TensorProto_DataType>(t_proto.data_type());
  const void* const raw_data = t_proto.has_raw_data() ? t_proto.raw_data().data() : nullptr;
  const size_t raw_data_len = t_proto.has_raw_data() ? t_proto.raw_data().size() : 0;
  switch (tensor_type_) {
    case TensorProto::BOOL:
      FETCH_VALUE_DATA(value_.ui64_, bool);
      break;
    case TensorProto::FLOAT:
      FETCH_VALUE_DATA(value_.fl_, float);
      break;
    case TensorProto::FLOAT16:
      FETCH_VALUE_DATA(value_.fl16_, MLFloat16);
      break;
    case TensorProto::DOUBLE:
      FETCH_VALUE_DATA(value_.dbl_, double);
      break;
    case TensorProto::INT8:
      FETCH_VALUE_DATA(value_.i64_, int8_t);
      break;
    case TensorProto::INT16:
      FETCH_VALUE_DATA(value_.i64_, int16_t);
      break;
    case TensorProto::INT32:
      FETCH_VALUE_DATA(value_.i64_, int32_t);
      break;
    case TensorProto::INT64:
      FETCH_VALUE_DATA(value_.i64_, int64_t);
      break;
    case TensorProto::UINT8:
      FETCH_VALUE_DATA(value_.ui64_, uint8_t);
      break;
    case TensorProto::UINT16:
      FETCH_VALUE_DATA(value_.ui64_, uint16_t);
      break;
    case TensorProto::UINT32:
      FETCH_VALUE_DATA(value_.ui64_, uint32_t);
      break;
    case TensorProto::UINT64:
      FETCH_VALUE_DATA(value_.ui64_, uint64_t);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", TensorProto::DataType_Name(tensor_type_));
      break;
  }
}

#undef FETCH_VALUE_DATA

template <class T>
inline T onnxruntime::ConstantOfShape::Value::GetFromSigned() const {
  return static_cast<T>(i64_);
}

template <class T>
inline T onnxruntime::ConstantOfShape::Value::GetFromUnsigned() const {
  return static_cast<T>(ui64_);
}

template <class T>
inline void FilloutOutput(T value, Tensor* output_tensor) {
  auto out = gsl::make_span(output_tensor->template MutableData<T>(), output_tensor->Shape().Size());
  std::fill(out.begin(), out.end(), value);
}

void onnxruntime::ConstantOfShape::DispatchTypeAndFillOutput(Tensor* output_tensor) const {
  switch (tensor_type_) {
    case TensorProto::BOOL:
      FilloutOutput(value_.GetFromUnsigned<bool>(), output_tensor);
      break;
    case TensorProto::FLOAT:
      FilloutOutput(value_.GetFloat(), output_tensor);
      break;
    case TensorProto::FLOAT16:
      FilloutOutput(value_.GetFloat16(), output_tensor);
      break;
    case TensorProto::DOUBLE:
      FilloutOutput(value_.GetDouble(), output_tensor);
      break;
    case TensorProto::INT8:
      FilloutOutput(value_.GetFromSigned<int8_t>(), output_tensor);
      break;
    case TensorProto::INT16:
      FilloutOutput(value_.GetFromSigned<int16_t>(), output_tensor);
      break;
    case TensorProto::INT32:
      FilloutOutput(value_.GetFromSigned<int32_t>(), output_tensor);
      break;
    case TensorProto::INT64:
      FilloutOutput(value_.GetFromSigned<int64_t>(), output_tensor);
      break;
    case TensorProto::UINT8:
      FilloutOutput(value_.GetFromUnsigned<uint8_t>(), output_tensor);
      break;
    case TensorProto::UINT16:
      FilloutOutput(value_.GetFromUnsigned<uint16_t>(), output_tensor);
      break;
    case TensorProto::UINT32:
      FilloutOutput(value_.GetFromUnsigned<uint32_t>(), output_tensor);
      break;
    case TensorProto::UINT64:
      FilloutOutput(value_.GetFromUnsigned<uint64_t>(), output_tensor);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", TensorProto::DataType_Name(tensor_type_));
      break;
  }
}

ConstantOfShape::ConstantOfShape(const OpKernelInfo& info) : OpKernel(info) {
  TensorProto t_proto;
  if (info.GetAttr<TensorProto>("value", &t_proto).IsOK()) {
    ORT_ENFORCE(t_proto.dims_size() == 1, "Must have a single dimension");
    ORT_ENFORCE(t_proto.dims()[0] == 1, "Must have a single dimension of 1");
    SetValue(t_proto);
  } else {
    tensor_type_ = TensorProto::FLOAT;
    value_.fl_ = 0.f;
  }
}

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {
  auto shape_tensor = ctx->Input<Tensor>(0);

  if (shape_tensor->DataType() != DataTypeImpl::GetType<int64_t>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input tensor expected to contain int64 data");
  }

  auto& input_shape = shape_tensor->Shape();

  // If empty the output is a scalar with empty shape
  // TensorShape::Size() will still return 1 and we will output
  // one value
  std::vector<int64_t> output_dims;
  if (input_shape.NumDimensions() > 0) {
    auto span = gsl::make_span(shape_tensor->Data<int64_t>(), input_shape.Size());
    output_dims.insert(output_dims.end(), span.cbegin(), span.cend());
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  DispatchTypeAndFillOutput(output_tensor);
  return Status::OK();
}
}  // namespace onnxruntime
