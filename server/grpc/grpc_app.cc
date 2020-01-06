// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "grpc_app.h"
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/channelz_service_plugin.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
namespace onnx_grpc = onnxruntime::server::grpc;

namespace onnxruntime {
namespace server {
GRPCApp::GRPCApp(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env, const std::string& host, const unsigned short port) : prediction_service_implementation_(env) {
  ::grpc::EnableDefaultHealthCheckService(true);
  ::grpc::channelz::experimental::InitChannelzService();
  ::grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&prediction_service_implementation_);
  builder.AddListeningPort(host + ":" + std::to_string(port), ::grpc::InsecureServerCredentials());

  server_ = builder.BuildAndStart();
  server_->GetHealthCheckService()->SetServingStatus(PredictionService::service_full_name(), true);
}

void GRPCApp::Run() {
  server_->Wait();
}
}  // namespace server
}  // namespace onnxruntime