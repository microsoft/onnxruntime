// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include "serializing/mem_buffer.h"
#include "serializing/tensorprotoutils.h"

#include "onnx-ml.pb.h"
#include "predict.pb.h"

#include "converter.h"
#include "executor.h"
#include "util.h"

namespace onnxruntime {
namespace server {

namespace protobufutil = google::protobuf::util;

protobufutil::Status Executor::SetMLValue(const onnx::TensorProto& input_tensor,
                                          MemBufferArray& buffers,
                                          OrtMemoryInfo* cpu_memory_info,
                                          /* out */ Ort::Value& ml_value) {
  auto logger = env_->GetLogger(request_id_);

  size_t cpu_tensor_length = 0;
  try {
    onnxruntime::server::GetSizeInBytesFromTensorProto<0>(input_tensor, &cpu_tensor_length);
  } catch (const Ort::Exception& e) {
    logger->error("GetSizeInBytesFromTensorProto() failed. Error Message: {}", e.what());
    return GenerateProtobufStatus(e.GetOrtErrorCode(), e.what());
  }

  auto* buf = buffers.AllocNewBuffer(cpu_tensor_length);
  try {
    onnxruntime::server::TensorProtoToMLValue(input_tensor,
                                              onnxruntime::server::MemBuffer(buf, cpu_tensor_length, *cpu_memory_info),
                                              ml_value);

  } catch (const Ort::Exception& e) {
    logger->error("TensorProtoToMLValue() failed. Message: {}", e.what());
    return GenerateProtobufStatus(e.GetOrtErrorCode(), e.what());
  }

  return protobufutil::Status::OK;
}

protobufutil::Status Executor::SetNameMLValueMap(std::vector<std::string>& input_names,
                                                 std::vector<Ort::Value>& input_values,
                                                 const onnxruntime::server::PredictRequest& request,
                                                 MemBufferArray& buffers) {
  auto logger = env_->GetLogger(request_id_);

  OrtMemoryInfo* memory_info = nullptr;
  auto ort_status = Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

  if (ort_status != nullptr || memory_info == nullptr) {
    logger->error("OrtCreateCpuMemoryInfo failed");
    return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "OrtCreateCpuMemoryInfo() failed");
  }

  // Prepare the Value object
  for (const auto& input : request.inputs()) {
    using_raw_data_ = using_raw_data_ && input.second.has_raw_data();

    Ort::Value ml_value{nullptr};
    auto status = SetMLValue(input.second, buffers, memory_info, ml_value);
    if (status != protobufutil::Status::OK) {
      Ort::GetApi().ReleaseMemoryInfo(memory_info);
      logger->error("SetMLValue() failed! Input name: {}", input.first);
      return status;
    }

    input_names.push_back(input.first);
    input_values.push_back(std::move(ml_value));
  }

  Ort::GetApi().ReleaseMemoryInfo(memory_info);
  return protobufutil::Status::OK;
}

std::vector<Ort::Value> Run(const Ort::Session& session, const Ort::RunOptions& options, const std::vector<std::string>& input_names, const std::vector<Ort::Value>& input_values, const std::vector<std::string>& output_names) {
  size_t input_count = input_names.size();
  size_t output_count = output_names.size();

  std::vector<const char*> input_ptrs{};
  input_ptrs.reserve(input_count);
  for (const auto& input : input_names) {
    input_ptrs.push_back(input.data());
  }
  std::vector<const char*> output_ptrs{};
  output_ptrs.reserve(output_count);
  for (const auto& output : output_names) {
    output_ptrs.push_back(output.data());
  }

  return const_cast<Ort::Session&>(session).Run(options, input_ptrs.data(), const_cast<Ort::Value*>(input_values.data()), input_count, output_ptrs.data(), output_count);
}

protobufutil::Status Executor::Predict(const std::string& model_name,
                                       const std::string& model_version,
                                       const onnxruntime::server::PredictRequest& request,
                                       /* out */ onnxruntime::server::PredictResponse& response) {
  auto logger = env_->GetLogger(request_id_);

  // Convert PredictRequest to NameMLValMap
  MemBufferArray buffer_array;
  std::vector<std::string> input_names;
  std::vector<Ort::Value> input_values;
  auto conversion_status = SetNameMLValueMap(input_names, input_values, request, buffer_array);
  if (conversion_status != protobufutil::Status::OK) {
    return conversion_status;
  }

  Ort::RunOptions run_options{};
  run_options.SetRunLogVerbosityLevel(static_cast<int>(env_->GetLogSeverity()));
  run_options.SetRunTag(request_id_.c_str());

  // Prepare the output names
  std::vector<std::string> output_names;

  if (!request.output_filter().empty()) {
    output_names.reserve(request.output_filter_size());
    for (const auto& name : request.output_filter()) {
      output_names.push_back(name);
    }
  } else {
    output_names = env_->GetModelOutputNames(model_name, model_version);
  }

  std::vector<Ort::Value> outputs;
  try {
    outputs = Run(env_->GetSession(model_name, model_version), run_options, input_names, input_values, output_names);
  } catch (const Ort::Exception& e) {
    return GenerateProtobufStatus(e.GetOrtErrorCode(), e.what());
  }

  // Build the response
  for (size_t i = 0, sz = outputs.size(); i < sz; ++i) {
    onnx::TensorProto output_tensor{};
    try {
      MLValueToTensorProto(outputs[i], using_raw_data_, logger, output_tensor);
    } catch (const Ort::Exception& e) {
      logger = env_->GetLogger(request_id_);
      logger->error("MLValueToTensorProto() failed. Output name: {}. Error Message: {}", output_names[i], e.what());
      return GenerateProtobufStatus(e.GetOrtErrorCode(), e.what());
    }

    auto insertion_result = response.mutable_outputs()->insert({output_names[i], output_tensor});

    if (!insertion_result.second) {
      logger->error("SetNameMLValueMap() failed. Output name: {}. Trying to overwrite existing output value", output_names[i]);
      return protobufutil::Status(protobufutil::error::Code::INVALID_ARGUMENT, "SetNameMLValueMap() failed: Cannot have two outputs with the same name");
    }
  }

  return protobufutil::Status::OK;
}

}  // namespace server
}  // namespace onnxruntime
