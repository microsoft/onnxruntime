#pragma once
#include <grpcpp/grpcpp.h>
#include "prediction_service_impl.h"
#include "environment.h"

namespace onnxruntime {
namespace server {
class GRPCApp {
 public:
  GRPCApp(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env, std::string host, const unsigned short port);
  ~GRPCApp() = default;
  GRPCApp(const GRPCApp& other) = delete;
  GRPCApp(GRPCApp&& other) = default;

  //Block until the server shuts down.
  void Run();

 private:
  grpc::PredictionServiceImpl prediction_service_implementation_;
  std::unique_ptr<::grpc::Server> server_;
};
}  // namespace server
}  // namespace onnxruntime