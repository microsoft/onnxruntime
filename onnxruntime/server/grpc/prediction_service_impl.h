#pragma once
#include "prediction_service.grpc.pb.h"
#include "../environment.h"
#include "../executor.h"
#include <grpcpp/grpcpp.h>

namespace onnxruntime {
namespace server {
namespace grpc {
class PredictionServiceImpl final : public onnxruntime::server::PredictionService::Service {
 public:
  PredictionServiceImpl(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env);

 private:
  std::shared_ptr<onnxruntime::server::ServerEnvironment> m_env;

  ::grpc::Status Predict(::grpc::ServerContext* context, const ::onnxruntime::server::PredictRequest* request, ::onnxruntime::server::PredictResponse* response);
};
}  // namespace grpc
}  // namespace server

}  // namespace onnxruntime