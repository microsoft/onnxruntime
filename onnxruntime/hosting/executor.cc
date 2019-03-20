// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

protobufutil::Status Executor::predict(const std::string& name, const std::string& version, const std::string& request_id,
                                       onnxruntime::hosting::PredictRequest& request,
                                       /* out */ onnxruntime::hosting::PredictResponse& response) {
  // Prepare MLValues
  std::unique_ptr<onnxruntime::MLValue> ml_value = std::make_unique<onnxruntime::MLValue>();
  std::unique_ptr<OrtCallback> del = std::make_unique<OrtCallback>();
  OrtAllocatorInfo* cpuAllocatorInfo;
  auto st = OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo);
  if (st != nullptr) {
    LOGS(env_.GetLogger(), ERROR) << "OrtCreateAllocatorInfo FAILED!";
  }
  auto one_tensorproto = request.inputs().begin()->second;
  size_t cpu_tensor_length;
  auto getsize_status = onnxruntime::utils::GetSizeInBytesFromTensorProto<0>(one_tensorproto, &cpu_tensor_length);
  std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
  auto status =
      onnxruntime::utils::TensorProtoToMLValue(onnxruntime::Env::Default(), nullptr, one_tensorproto,
                                               onnxruntime::MemBuffer(data.get(), cpu_tensor_length, *cpuAllocatorInfo),
                                               *ml_value, *del);

  // Prepare for Run()
  OrtRunOptions runOptions{};
  runOptions.run_log_verbosity_level = 4;
  runOptions.run_tag = request_id;
  onnxruntime::NameMLValMap nameMlValMap;
  nameMlValMap[request.inputs().begin()->first] = *ml_value;
  std::vector<std::string> output_names{request.output_filter(0)};
  std::vector<onnxruntime::MLValue> outputs;

  // Run()
  auto run_status = env_.GetSession()->Run(runOptions, nameMlValMap, output_names, &outputs);

  // Prepare return value
  onnx::TensorProto output_tensor;
  auto mlvalue2tensorproto_status = MLValue2TensorProto(outputs[0], one_tensorproto.has_raw_data(), output_tensor);

  response.mutable_outputs()->insert({output_names[0], output_tensor});
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

  // One time of data type mapping activity has limit information to us.
  // But the collection of this information will let us know the frequency of data types.
  // Above if-statement order could be optimized with the statistic.
  LOGS(env_.GetLogger(), VERBOSE) << "Converted TensorProto_DataType: " << type;
  return type;
}

protobufutil::Status Executor::MLValue2TensorProto(onnxruntime::MLValue& ml_value, bool using_raw_data, /* out */ onnx::TensorProto& tensor_proto) {
  // Tensor in MLValue
  auto* tensor = ml_value.GetMutable<onnxruntime::Tensor>();

  // dims
  const onnxruntime::TensorShape& tensor_shape = tensor->Shape();
  for (auto dim : tensor_shape.GetDims()) {
    tensor_proto.add_dims(dim);
  }

  // data_type
  auto data_type = MLDataTypeToTensorProtoDataType(tensor->DataType());
  tensor_proto.set_data_type(data_type);

  // segment: ignored for now. We do not expect very large tensors in the output

  // data
  if (using_raw_data) {
    tensor_proto.set_raw_data(tensor->Data<float>(), tensor->Size());
  }

  return protobufutil::Status(protobufutil::Status::OK);
}

}  // namespace hosting
}  // namespace onnxruntime