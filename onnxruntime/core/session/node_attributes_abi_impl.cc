// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/status.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include "core/graph/basic_types.h"
#include "core/framework/data_types.h"

ORT_API_STATUS_IMPL(OrtApis::CreateNodeAttributes, _Outptr_ OrtNodeAttributes** attributes_) {
  onnxruntime::NodeAttributes* attributes = new onnxruntime::NodeAttributes;
  *attributes_ = reinterpret_cast<OrtNodeAttributes*>(attributes);
  return onnxruntime::ToOrtStatus(onnxruntime::Status::OK());
}

#define ATTR_SETTER_IMPL(ctype, apiName, enumType, field)                                                  \
  ORT_API_STATUS_IMPL(OrtApis::NodeAttributes_Set_##apiName,                                               \
                      _Inout_ OrtNodeAttributes* attributes_,                                              \
                      _In_ const char* name,                                                               \
                      ctype value) {                                                                       \
    API_IMPL_BEGIN                                                                                         \
    ORT_ENFORCE(attributes_, "attributes must be non-null.");                                              \
    onnxruntime::NodeAttributes* attributes = reinterpret_cast<onnxruntime::NodeAttributes*>(attributes_); \
    ONNX_NAMESPACE::AttributeProto attribute_proto;                                                        \
    attribute_proto.set_name(name);                                                                        \
    attribute_proto.set_type(enumType);                                                                    \
    attribute_proto.set_##field(value);                                                                    \
    attributes->insert(std::make_pair(name, attribute_proto));                                             \
    return onnxruntime::ToOrtStatus(onnxruntime::Status::OK());                                            \
    API_IMPL_END                                                                                           \
  }

#define ARRAY_ATTR_SETTER_IMPL(ctype, apiName, enumType, field)                                            \
  ORT_API_STATUS_IMPL(OrtApis::NodeAttributes_SetArray_##apiName,                                          \
                      _Inout_ OrtNodeAttributes* attributes_,                                              \
                      _In_ const char* name,                                                               \
                      _In_ ctype const* values,                                                            \
                      size_t num_values) {                                                                 \
    API_IMPL_BEGIN                                                                                         \
    ORT_ENFORCE(attributes_, "attributes must be non-null.");                                              \
    onnxruntime::NodeAttributes* attributes = reinterpret_cast<onnxruntime::NodeAttributes*>(attributes_); \
    ONNX_NAMESPACE::AttributeProto attribute_proto;                                                        \
    attribute_proto.set_name(name);                                                                        \
    attribute_proto.set_type(enumType);                                                                    \
    for (size_t i = 0; i < num_values; i++) {                                                              \
      attribute_proto.add_##field(values[i]);                                                              \
    }                                                                                                      \
    attributes->insert(std::make_pair(name, attribute_proto));                                             \
    return onnxruntime::ToOrtStatus(onnxruntime::Status::OK());                                            \
    API_IMPL_END                                                                                           \
  }

ATTR_SETTER_IMPL(const char*, string, ONNX_NAMESPACE::AttributeProto::STRING, s)
ATTR_SETTER_IMPL(float, float, ONNX_NAMESPACE::AttributeProto::FLOAT, f)
ATTR_SETTER_IMPL(int64_t, int64, ONNX_NAMESPACE::AttributeProto::INT, i)
ARRAY_ATTR_SETTER_IMPL(float, float, ONNX_NAMESPACE::AttributeProto::FLOATS, floats)
ARRAY_ATTR_SETTER_IMPL(int64_t, int64, ONNX_NAMESPACE::AttributeProto::INTS, ints)
ARRAY_ATTR_SETTER_IMPL(const char*, string, ONNX_NAMESPACE::AttributeProto::STRINGS, strings)

ORT_API_STATUS_IMPL(OrtApis::NodeAttributes_Set_tensor,
                    _Inout_ OrtNodeAttributes* attributes_,
                    _In_ const char* name,
                    _Inout_ void* p_data,
                    size_t p_data_len,
                    _In_ const int64_t* shape,
                    size_t shape_len,
                    ONNXTensorElementDataType type) {
  API_IMPL_BEGIN
  ORT_ENFORCE(attributes_, "attributes must be non-null.");
  onnxruntime::NodeAttributes* attributes = reinterpret_cast<onnxruntime::NodeAttributes*>(attributes_);

  ONNX_NAMESPACE::AttributeProto attribute_proto;

  attribute_proto.set_name(name);
  attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  ONNX_NAMESPACE::TensorProto* t = attribute_proto.mutable_t();

  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT8);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::INT8);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT16);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::INT16);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::INT32);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT32);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT64);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::BOOL);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      t->set_data_type(ONNX_NAMESPACE::TensorProto::DOUBLE);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return onnxruntime::ToOrtStatus(onnxruntime::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, errmsg));
    }
  }

  for (size_t i = 0; i < shape_len; i++) {
    t->add_dims(shape[i]);
  }

  for (size_t i = 0; i < p_data_len; i++) {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        t->add_float_data(static_cast<float*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        t->add_int32_data(static_cast<uint8_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        t->add_int32_data(static_cast<int8_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        t->add_int32_data(static_cast<uint16_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        t->add_int32_data(static_cast<bool*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        t->add_int32_data(static_cast<int16_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        t->add_int32_data(static_cast<int32_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        t->add_int64_data(static_cast<int64_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        t->add_uint64_data(static_cast<uint32_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        t->add_uint64_data(static_cast<uint64_t*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        t->add_double_data(static_cast<double*>(p_data)[i]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      default: {
        std::ostringstream oss;
        oss << "type " << type << " is not supported in this function";
        std::string errmsg = oss.str();
        return onnxruntime::ToOrtStatus(onnxruntime::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, errmsg));
      }
    }
  }

  attributes->insert(std::make_pair(name, attribute_proto));
  return onnxruntime::ToOrtStatus(onnxruntime::Status::OK());
  API_IMPL_END
}
