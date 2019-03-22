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

#include "executor.h"

namespace onnxruntime {
namespace hosting {

namespace protobufutil = google::protobuf::util;

// TODO: make all logging has request id

protobufutil::Status Executor::predict(const std::string& name, const std::string& version, const std::string& request_id,
                                       onnxruntime::hosting::PredictRequest& request,
                                       /* out */ onnxruntime::hosting::PredictResponse& response) {
  // If any input data is in raw_data field, the output will be put into raw_data field.
  bool using_raw_data = true;

  // Prepare the input NameMLValMap
  onnxruntime::NameMLValMap nameMlValMap;
  common::Status status;
  for (const auto& input : request.inputs()) {
    std::string input_name = input.first;
    onnx::TensorProto input_tensor = input.second;
    using_raw_data = using_raw_data && input_tensor.has_raw_data();

    // Prepare the MLValue
    OrtAllocatorInfo* cpuAllocatorInfo;
    auto ort_status = OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo);
    if (ort_status != nullptr || cpuAllocatorInfo == nullptr) {
      LOGS(env_.GetLogger(), ERROR) << "OrtCreateAllocatorInfo FAILED! Input name: " << input_name;
      return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "OrtCreateAllocatorInfo() FAILED!");
    }

    size_t cpu_tensor_length;
    status = onnxruntime::utils::GetSizeInBytesFromTensorProto<0>(input_tensor, &cpu_tensor_length);
    if (!status.IsOK()) {
      LOGS(env_.GetLogger(), ERROR) << "GetSizeInBytesFromTensorProto() FAILED! Input name: " << input_name
                                    << ". Error code: " << status.Code()
                                    << ". Error Message: " << status.ErrorMessage();
      return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "GetSizeInBytesFromTensorProto() FAILED!");
    }

    std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
    if (nullptr == data) {
      LOGS(env_.GetLogger(), ERROR) << "Run out memory. Input name: " << input_name;
      return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "Run out of memory");
    }

    MLValue ml_value;
    OrtCallback deleter;
    status = onnxruntime::utils::TensorProtoToMLValue(onnxruntime::Env::Default(), nullptr, input_tensor,
                                                      onnxruntime::MemBuffer(data.get(), cpu_tensor_length, *cpuAllocatorInfo),
                                                      ml_value, deleter);
    if (!status.IsOK()) {
      LOGS(env_.GetLogger(), ERROR) << "TensorProtoToMLValue() FAILED! Input name: " << input_name
                                    << ". Error code: " << status.Code()
                                    << ". Error Message: " << status.ErrorMessage();
      return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "TensorProtoToMLValue() FAILED!");
    }

    nameMlValMap[input_name] = ml_value;
  }  // for(const auto& input : request.inputs())

  // Prepare output names
  std::vector<std::string> output_names;
  for (auto name : request.output_filter()) {
    output_names.push_back(name);
  }

  // Output MLValue vector
  std::vector<onnxruntime::MLValue> outputs(output_names.size());

  // Run()!
  OrtRunOptions runOptions{};
  runOptions.run_log_verbosity_level = 4;  // TODO: respect user selected log level
  runOptions.run_tag = request_id;

  status = env_.GetSession()->Run(runOptions, nameMlValMap, output_names, &outputs);
  if (!status.IsOK()) {
    LOGS(env_.GetLogger(), ERROR) << "Run() FAILED!"
                                  << ". Error code: " << status.Code()
                                  << ". Error Message: " << status.ErrorMessage();
    return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "Run() FAILED!");
  }

  // Prepare for response
  for (size_t i = 0; i < outputs.size(); ++i) {
    onnx::TensorProto output_tensor;
    status = MLValue2TensorProto(outputs[i], using_raw_data, output_tensor);
    if (!status.IsOK()) {
      LOGS(env_.GetLogger(), ERROR) << "MLValue2TensorProto() FAILED! Output name: " << output_names[i]
                                    << ". Error code: " << status.Code()
                                    << ". Error Message: " << status.ErrorMessage();
      return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "MLValue2TensorProto() FAILED!");
    }

    response.mutable_outputs()->insert({output_names[i], output_tensor});
  }

  return google::protobuf::util::Status::OK;
}

onnx::TensorProto_DataType Executor::MLDataTypeToTensorProtoDataType(const onnxruntime::DataTypeImpl* cpp_type) {
  onnx::TensorProto_DataType type;
  if (cpp_type == onnxruntime::DataTypeImpl::GetType<float>()) {
    type = onnx::TensorProto_DataType_FLOAT;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint8_t>()) {
    type = onnx::TensorProto_DataType_UINT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int8_t>()) {
    type = onnx::TensorProto_DataType_INT8;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint16_t>()) {
    type = onnx::TensorProto_DataType_UINT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int16_t>()) {
    type = onnx::TensorProto_DataType_INT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int32_t>()) {
    type = onnx::TensorProto_DataType_INT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<int64_t>()) {
    type = onnx::TensorProto_DataType_INT64;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<std::string>()) {
    type = onnx::TensorProto_DataType_STRING;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<bool>()) {
    type = onnx::TensorProto_DataType_BOOL;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>()) {
    type = onnx::TensorProto_DataType_FLOAT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>()) {
    type = onnx::TensorProto_DataType_BFLOAT16;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<double>()) {
    type = onnx::TensorProto_DataType_DOUBLE;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint32_t>()) {
    type = onnx::TensorProto_DataType_UINT32;
  } else if (cpp_type == onnxruntime::DataTypeImpl::GetType<uint64_t>()) {
    type = onnx::TensorProto_DataType_UINT64;
  } else {
    type = onnx::TensorProto_DataType_UNDEFINED;
  }

  // One time of data type mapping activity usage has limit information to us.
  // But the collection of this information will let us know the frequency of data types.
  // Above if-statement order could be optimized with the statistic.
  LOGS(env_.GetLogger(), VERBOSE) << "Converted TensorProto_DataType: " << type;
  return type;
}

common::Status Executor::MLValue2TensorProto(onnxruntime::MLValue& ml_value, bool using_raw_data, /* out */
                                             onnx::TensorProto& tensor_proto) {
  // Tensor in MLValue
  auto* tensor = ml_value.GetMutable<onnxruntime::Tensor>();

  // dims field
  const onnxruntime::TensorShape& tensor_shape = tensor->Shape();
  for (auto dim : tensor_shape.GetDims()) {
    tensor_proto.add_dims(dim);
  }

  // data_type field
  onnx::TensorProto_DataType data_type = MLDataTypeToTensorProtoDataType(tensor->DataType());
  tensor_proto.set_data_type(data_type);

  // data_location field: Data is stored in raw_data (if set) otherwise in type-specified field.
  if (using_raw_data) {
    tensor_proto.set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
  }

  // *_data field
  // According to onnx_ml.proto, depending on the data_type field,
  // exactly one of the *_data fields is used to store the elements of the tensor.
  switch (data_type) {
    case onnx::TensorProto_DataType_FLOAT: {
      auto data = tensor->Data<float>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor->Size());
      } else {
        size_t data_length = tensor->Size() / sizeof(float);
        for (size_t i = 0; i < data_length; ++i) {
          tensor_proto.add_float_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_INT32:
    case onnx::TensorProto_DataType_UINT8:
    case onnx::TensorProto_DataType_INT8:
    case onnx::TensorProto_DataType_UINT16:
    case onnx::TensorProto_DataType_INT16:
    case onnx::TensorProto_DataType_BOOL:
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_BFLOAT16: {
      // TODO: special handle FLOAT16 and BFLOAT16?
      auto data = tensor->Data<int32_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor->Size());
      } else {
        size_t data_length = tensor->Size() / sizeof(int32_t);
        for (size_t i = 0; i < data_length; ++i) {
          tensor_proto.add_int32_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_STRING: {
      // string could not be written into "raw_data"
      auto data = tensor->Data<int32_t>();
      size_t data_length = tensor->Size() / sizeof(int32_t);
      for (size_t i = 0; i < data_length; ++i) {
        tensor_proto.add_int32_data(data[i]);
      }
      break;
    }
    case onnx::TensorProto_DataType_INT64: {
      auto data = tensor->Data<int64_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor->Size());
      } else {
        size_t data_length = tensor->Size() / sizeof(int64_t);
        for (size_t i = 0; i < data_length; ++i) {
          tensor_proto.add_int64_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT64: {
      auto data = tensor->Data<uint64_t>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor->Size());
      } else {
        size_t data_length = tensor->Size() / sizeof(uint64_t);
        for (size_t i = 0; i < data_length; ++i) {
          tensor_proto.add_uint64_data(data[i]);
        }
      }
      break;
    }
    case onnx::TensorProto_DataType_DOUBLE: {
      auto data = tensor->Data<double>();
      if (using_raw_data) {
        tensor_proto.set_raw_data(data, tensor->Size());
      } else {
        size_t data_length = tensor->Size() / sizeof(double);
        for (size_t i = 0; i < data_length; ++i) {
          tensor_proto.add_double_data(data[i]);
        }
      }
      break;
    }
    default: {
      LOGS(env_.GetLogger(), ERROR) << "Unsupported TensorProto DataType: " << data_type;
      return common::Status(common::StatusCategory::ONNXRUNTIME,
                            common::StatusCode::NOT_IMPLEMENTED,
                            "Unsupported TensorProto DataType: " + std::to_string(data_type));
    }
  }

  return common::Status::OK();
}

}  // namespace hosting
}  // namespace onnxruntime