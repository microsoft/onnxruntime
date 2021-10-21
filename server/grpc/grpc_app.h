// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <grpcpp/grpcpp.h>
#include "prediction_service_impl.h"
#include "environment.h"

namespace onnxruntime {
namespace server {
class GRPCApp {
 public:
  GRPCApp(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env, const std::string& host, const unsigned short port);
  ~GRPCApp() = default;
  GRPCApp(const GRPCApp& other) = delete;
  GRPCApp(GRPCApp&& other) = delete;

  GRPCApp& operator=(const GRPCApp&) = delete; 

  //Block until the server shuts down.
  void Run();

 private:
  grpc::PredictionServiceImpl prediction_service_implementation_;
  std::unique_ptr<::grpc::Server> server_;
};
}  // namespace server
}  // namespace onnxruntime