#pragma once
#include <grpcpp/grpcpp.h>
#include "prediction_service_impl.h"
#include "../environment.h"

namespace onnxruntime {
namespace server {
class GRPCApp {
 public:
  GRPCApp(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env, std::string host, const unsigned short port);
  ~GRPCApp() = default;
  GRPCApp(GRPCApp const& other) = delete;
  GRPCApp(GRPCApp&& other) = default;

  //Block until the server shuts down.
  void Run();

 private:
  grpc::PredictionServiceImpl m_service;
  std::unique_ptr<::grpc::Server> m_server;
};
}  // namespace server
}  // namespace onnxruntime