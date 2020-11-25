// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "prediction_service.grpc.pb.h"
#include "environment.h"
#include "executor.h"
#include <grpcpp/grpcpp.h>

namespace onnxruntime {
namespace server {
namespace grpc {
class PredictionServiceImpl final : public onnxruntime::server::PredictionService::Service {
 public:
  PredictionServiceImpl(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env);
  ::grpc::Status Predict(::grpc::ServerContext* context, const ::onnxruntime::server::PredictRequest* request, ::onnxruntime::server::PredictResponse* response);

 private:
  std::shared_ptr<onnxruntime::server::ServerEnvironment> environment_;

  //Extract customer request ID and set request ID for response.
  std::string SetRequestContext(::grpc::ServerContext* context);
};
}  // namespace grpc
}  // namespace server

}  // namespace onnxruntime