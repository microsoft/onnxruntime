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
#include "executor.h"

namespace onnxruntime {
namespace hosting {

namespace protobufutil = google::protobuf::util;

// TODO: make all logging has request id

protobufutil::Status Executor::Predict(const std::string& name, const std::string& version, const std::string& request_id,
                                       onnxruntime::hosting::PredictRequest& request,
                                       /* out */ onnxruntime::hosting::PredictResponse& response) {
  bool using_raw_data = true;
  auto logger = env_->GetLogger();

  // Create the input NameMLValMap
  onnxruntime::NameMLValMap nameMlValMap{};
  common::Status status{};
  for (const auto& input : request.inputs()) {
    std::string input_name = input.first;
    onnx::TensorProto input_tensor = input.second;
    using_raw_data = using_raw_data && input_tensor.has_raw_data();

    // Prepare the MLValue object
    OrtAllocatorInfo* cpuAllocatorInfo;
    auto ort_status = OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpuAllocatorInfo);
    if (ort_status != nullptr || cpuAllocatorInfo == nullptr) {
      LOGS(logger, ERROR) << "OrtCreateAllocatorInfo FAILED! Input name: " << input_name;
      return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "OrtCreateAllocatorInfo() FAILED!");
    }

    size_t cpu_tensor_length;
    status = onnxruntime::utils::GetSizeInBytesFromTensorProto<0>(input_tensor, &cpu_tensor_length);
    if (!status.IsOK()) {
      LOGS(logger, ERROR) << "GetSizeInBytesFromTensorProto() FAILED! Input name: " << input_name
                          << ". Error code: " << status.Code()
                          << ". Error Message: " << status.ErrorMessage();
      return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "GetSizeInBytesFromTensorProto() FAILED!");
    }

    std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
    if (nullptr == data) {
      LOGS(logger, ERROR) << "Run out memory. Input name: " << input_name;
      return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "Run out of memory");
    }

    // TensorProto -> MLValue
    MLValue ml_value;
    OrtCallback deleter;
    status = onnxruntime::utils::TensorProtoToMLValue(onnxruntime::Env::Default(), nullptr, input_tensor,
                                                      onnxruntime::MemBuffer(data.get(), cpu_tensor_length, *cpuAllocatorInfo),
                                                      ml_value, deleter);
    if (!status.IsOK()) {
      LOGS(logger, ERROR) << "TensorProtoToMLValue() FAILED! Input name: " << input_name
                          << ". Error code: " << status.Code()
                          << ". Error Message: " << status.ErrorMessage();
      return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "TensorProtoToMLValue() FAILED!");
    }

    nameMlValMap[input_name] = ml_value;
  }  // for(const auto& input : request.inputs())

  // Prepare the output names and vector
  std::vector<std::string> output_names;
  for (auto name : request.output_filter()) {
    output_names.push_back(name);
  }
  std::vector<onnxruntime::MLValue> outputs(output_names.size());

  // Run()!
  OrtRunOptions runOptions{};
  runOptions.run_log_verbosity_level = 4;  // TODO: respect user selected log level
  runOptions.run_tag = request_id;

  status = env_->GetSession()->Run(runOptions, nameMlValMap, output_names, &outputs);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Run() FAILED!"
                        << ". Error code: " << status.Code()
                        << ". Error Message: " << status.ErrorMessage();
    return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "Run() FAILED!");
  }

  // Build the response
  for (size_t i = 0; i < outputs.size(); ++i) {
    onnx::TensorProto output_tensor;
    status = MLValue2TensorProto(outputs[i], using_raw_data, logger, output_tensor);
    if (!status.IsOK()) {
      LOGS(logger, ERROR) << "MLValue2TensorProto() FAILED! Output name: " << output_names[i]
                          << ". Error code: " << status.Code()
                          << ". Error Message: " << status.ErrorMessage();
      return protobufutil::Status(static_cast<protobufutil::error::Code>(status.Code()), "MLValue2TensorProto() FAILED!");
    }

    response.mutable_outputs()->insert({output_names[i], output_tensor});
  }

  return protobufutil::Status::OK;
}

}  // namespace hosting
}  // namespace onnxruntime