// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/environment.h"
#include "core/framework/framework_common.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"

#include "onnx-ml.pb.h"
#include "predict.pb.h"

#include "converter.h"

namespace onnxruntime {
namespace hosting {

namespace protobufutil = google::protobuf::util;

onnx::TensorProto_DataType MLDataTypeToTensorProtoDataType(const onnxruntime::DataTypeImpl* cpp_type) {
  if (cpp_type == onnxruntime::DataTypeImpl::GetType<float>()) {
    return onnx::TensorProto_DataType_FLOAT;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint8_t>()) {
    return onnx::TensorProto_DataType_UINT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int8_t>()) {
    return onnx::TensorProto_DataType_INT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint16_t>()) {
    return onnx::TensorProto_DataType_UINT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int16_t>()) {
    return onnx::TensorProto_DataType_INT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int32_t>()) {
    return onnx::TensorProto_DataType_INT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int64_t>()) {
    return onnx::TensorProto_DataType_INT64;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<std::string>()) {
    return onnx::TensorProto_DataType_STRING;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<bool>()) {
    return onnx::TensorProto_DataType_BOOL;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>()) {
    return onnx::TensorProto_DataType_FLOAT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>()) {
    return onnx::TensorProto_DataType_BFLOAT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<double>()) {
    return onnx::TensorProto_DataType_DOUBLE;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint32_t>()) {
    return onnx::TensorProto_DataType_UINT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint64_t>()) {
    return onnx::TensorProto_DataType_UINT64;
  } else {
    return onnx::TensorProto_DataType_UNDEFINED;
  }
}

common::Status MLValueToTensorProto(onnxruntime::MLValue& ml_value, bool using_raw_data,
                                    std::unique_ptr<onnxruntime::logging::Logger> logger,
                                    /* out */ onnx::TensorProto& tensor_proto) {
  // Tensor in MLValue
  const auto& tensor = ml_value.Get<onnxruntime::Tensor>();

  // dims field
  const onnxruntime::TensorShape& tensor_shape = tensor.Shape();
  for (const auto& dim : tensor_shape.GetDims()) {
    tensor_proto.add_dims(dim);
  }

  // data_type field
  onnx::TensorProto_DataType data_type = MLDataTypeToTensorProtoDataType(tensor.DataType());
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
      const auto* data = tensor.Data<float>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        for (int i = 0, count = tensor.Shape().Size(); i < count; ++i) {
          tensor_proto.add_float_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT32: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<int32_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        for (int i = 0, count = tensor.Shape().Size(); i < count; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT8: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<uint8_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT8: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<int8_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT16: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<uint16_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT16: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<int16_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_BOOL: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<bool>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT16: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<onnxruntime::MLFloat16>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_BFLOAT16: {  // Target: raw_data or int32_data
      const auto* data = tensor.Data<onnxruntime::BFloat16>();
      const auto raw_data_size = tensor.Shape().Size();

      std::vector<uint16_t> raw_data;
      raw_data.reserve(raw_data_size);
      for (int i = 0; i < raw_data_size; ++i) {
        raw_data.push_back(data[i].val);
      }

      if (using_raw_data) {
        tensor_proto.set_raw_data(raw_data.data(), raw_data.size() * sizeof(uint16_t));
      } else {
        auto i32data = reinterpret_cast<const int32_t*>(raw_data.data());
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(int32_t)); i < count; ++i) {
          tensor_proto.add_int32_data(i32data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_STRING: {  // Target: string_data
      // string could not be written into "raw_data"
      const auto* data = tensor.Data<std::string>();
      for (int i = 0, count = tensor.Shape().Size(); i < count; ++i) {
        tensor_proto.add_string_data(data[i]);
      }
      break;
    }
    case onnx::TensorProto_DataType_INT64: {  // Target: raw_data or int64_data
      const auto* data = tensor.Data<int64_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        for (int x = 0, loop_length = tensor.Shape().Size(); x < loop_length; ++x) {
          tensor_proto.add_int64_data(data[x]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT32: {  // Target: raw_data or uint64_data
      const auto* data = tensor.Data<uint32_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        auto u64data = reinterpret_cast<const uint64_t*>(data);
        for (int i = 0, count = 1 + ((tensor.Size() - 1) / sizeof(uint64_t)); i < count; ++i) {
          tensor_proto.add_uint64_data(u64data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT64: {  // Target: raw_data or uint64_data
      const auto* data = tensor.Data<uint64_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        for (int x = 0, loop_length = tensor.Shape().Size(); x < loop_length; ++x) {
          tensor_proto.add_uint64_data(data[x]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_DOUBLE: {  // Target: raw_data or double_data
      auto data = tensor.Data<double>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor.Size());
      } else {
        for (int x = 0, loop_length = tensor.Shape().Size(); x < loop_length; ++x) {
          tensor_proto.add_double_data(data[x]);
        }
      }
      break;
    }
    default: {
      LOGS(*logger, ERROR) << "Unsupported TensorProto DataType: " << data_type;
      return common::Status(common::StatusCategory::ONNXRUNTIME,
                            common::StatusCode::NOT_IMPLEMENTED,
                            "Unsupported TensorProto DataType: " + std::to_string(data_type));
    }
  }

  return common::Status::OK();
}
}  // namespace hosting
}  // namespace onnxruntime