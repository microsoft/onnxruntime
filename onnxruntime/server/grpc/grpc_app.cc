#include "grpc_app.h"
#include <grpcpp/health_check_service_interface.h>

namespace onnx_grpc = onnxruntime::server::grpc;

namespace onnxruntime {
namespace server {
GRPCApp::GRPCApp(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env, std::string host, const unsigned short port) : m_service(env) {
  m_server = std::unique_ptr<::grpc::Server>(nullptr);
  ::grpc::EnableDefaultHealthCheckService(true);
  ::grpc::ServerBuilder builder;
  std::stringstream ss;
  ss << host << ":" << port;
  builder.RegisterService(&m_service);
  builder.AddListeningPort(ss.str(), ::grpc::InsecureServerCredentials());
  auto server = builder.BuildAndStart();
  m_server.swap(server);
  m_server->GetHealthCheckService()->SetServingStatus(PredictionService::service_full_name(), true);
}
}  // namespace server
}  // namespace onnxruntime