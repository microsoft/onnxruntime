// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>
#include "core/session/onnxruntime_cxx_api.h"


#include "onnx-ml.pb.h"
#include "predict.pb.h"

#include "converter.h"
#include "serializing/mem_buffer.h"

namespace onnxruntime {
namespace server {

namespace protobufutil = google::protobuf::util;

onnx::TensorProto_DataType MLDataTypeToTensorProtoDataType(ONNXTensorElementDataType cpp_type) {
  switch (cpp_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT8;  // maps to c type uint8_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT8;  // maps to c type int8_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT16;  // maps to c type uint16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT16;  // maps to c type int16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT32;  // maps to c type int32_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT64;  // maps to c type int64_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return onnx::TensorProto_DataType::TensorProto_DataType_STRING;  // maps to c++ type std::string
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return onnx::TensorProto_DataType::TensorProto_DataType_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE;  // maps to c type double
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT32;  // maps to c type uint32_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT64;  // maps to c type uint64_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64;  // complex with float32 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128;  // complex with float64 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
      return onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED;
  }
}

common::Status MLValueToTensorProto(Ort::Value& ml_value, bool using_raw_data,
                                    std::shared_ptr<spdlog::logger> logger,
                                    /* out */ onnx::TensorProto& tensor_proto) {
  if (!ml_value.IsTensor()) {
    //TODO: Throw?
    return common::Status(common::StatusCategory::ONNXRUNTIME,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "Don't support Non-Tensor values");
  }
  // Tensor in MLValue
  const auto& shape = ml_value.GetTensorTypeAndShapeInfo();

  // dims field
  for (const auto& dim : shape.GetShape()) {
    tensor_proto.add_dims(dim);
  }
  auto elem_count = shape.GetElementCount();

  // data_type field
  onnx::TensorProto_DataType data_type = MLDataTypeToTensorProtoDataType(shape.GetElementType());
  tensor_proto.set_data_type(data_type);

  // data_location field: Data is stored in raw_data (if set) otherwise in type-specified field.
  if (using_raw_data && data_type != onnx::TensorProto_DataType_STRING) {
    tensor_proto.set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
  }

  // *_data field
  // According to onnx_ml.proto, depending on the data_type field,
  // exactly one of the *_data fields is used to store the elements of the tensor.
  switch (data_type) {
    case onnx::TensorProto_DataType_FLOAT: {  // Target: raw_data or float_data
      const auto* data = ml_value.GetTensorMutableData<float>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(float) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_float_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT32: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<int32_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(int32_t) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT8: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<uint8_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(uint8_t) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT8: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<int8_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(int8_t) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT16: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<uint16_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(uint16_t) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT16: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<int16_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(int16_t) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_BOOL: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<bool>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(bool) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT16: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<onnxruntime::MLFloat16>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(onnxruntime::MLFloat16) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(reinterpret_cast<const uint16_t*>(data)[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_BFLOAT16: {  // Target: raw_data or int32_data
      const auto* data = ml_value.GetTensorMutableData<onnxruntime::BFloat16>();

      std::vector<uint16_t> raw_data;
      raw_data.reserve(elem_count);
      for (size_t i = 0; i < elem_count; ++i) {
        raw_data.push_back(data[i].val);
      }

      if (using_raw_data) {
        tensor_proto.set_raw_data(raw_data.data(), raw_data.size() * sizeof(uint16_t));
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_int32_data(raw_data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_STRING: {  // Target: string_data
      // string could not be written into "raw_data"
      auto length = ml_value.GetStringTensorDataLength();
      std::vector<char> buffer;
      std::vector<size_t> offsets;
      buffer.resize(length);
      offsets.resize(elem_count);
      ml_value.GetStringTensorContent(buffer.data(), length, offsets.data(), elem_count);
      size_t start = 0;
      for (size_t i = 1; i < elem_count; ++i) {
        auto end = offsets[i];
        tensor_proto.add_string_data(&buffer[start], end - start);
        start = end;
      }
      tensor_proto.add_string_data(&buffer[start], length - start);
      break;
    }
    case onnx::TensorProto_DataType_INT64: {  // Target: raw_data or int64_data
      const auto* data = ml_value.GetTensorMutableData<int64_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(int64_t) * elem_count);
      } else {
        for (size_t x = 0, loop_length = elem_count; x < loop_length; ++x) {
          tensor_proto.add_int64_data(data[x]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT32: {  // Target: raw_data or uint64_data
      const auto* data = ml_value.GetTensorMutableData<uint32_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(u_int32_t) * elem_count);
      } else {
        for (size_t i = 0, count = elem_count; i < count; ++i) {
          tensor_proto.add_uint64_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT64: {  // Target: raw_data or uint64_data
      const auto* data = ml_value.GetTensorMutableData<uint64_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(uint64_t) * elem_count);
      } else {
        for (size_t x = 0, loop_length = elem_count; x < loop_length; ++x) {
          tensor_proto.add_uint64_data(data[x]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_DOUBLE: {  // Target: raw_data or double_data
      auto data = ml_value.GetTensorMutableData<double>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, sizeof(double) * elem_count);
      } else {
        for (size_t x = 0, loop_length = elem_count; x < loop_length; ++x) {
          tensor_proto.add_double_data(data[x]);
        }
      }
      break;
    }
    default: {
      logger->error("Unsupported TensorProto DataType: {}", data_type);
      return common::Status(common::StatusCategory::ONNXRUNTIME,
                            common::StatusCode::NOT_IMPLEMENTED,
                            "Unsupported TensorProto DataType: " + std::to_string(data_type));
    }
  }

  return common::Status::OK();
}
}  // namespace server
}  // namespace onnxruntime