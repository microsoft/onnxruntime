// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <google/protobuf/stubs/status.h>

#include "environment.h"
#include "predict.pb.h"

namespace onnxruntime {
namespace server {

class Executor {
 public:
  Executor(ServerEnvironment* server_env, std::string request_id) : env_(server_env),
                                                                                      request_id_(std::move(request_id)),
                                                                                      using_raw_data_(true) {}

  // Prediction method
  google::protobuf::util::Status Predict(const std::string& model_name,
                                         const std::string& model_version,
                                         onnxruntime::server::PredictRequest& request,
                                         /* out */ onnxruntime::server::PredictResponse& response);

 private:
  ServerEnvironment* env_;
  const std::string request_id_;
  bool using_raw_data_;

  google::protobuf::util::Status SetMLValue(const onnx::TensorProto& input_tensor,
                                            OrtAllocatorInfo* cpu_allocator_info,
                                            /* out */ MLValue& ml_value);

  google::protobuf::util::Status SetNameMLValueMap(onnxruntime::NameMLValMap& name_value_map, const onnxruntime::server::PredictRequest& request);
};

}  // namespace server
}  // namespace onnxruntime
