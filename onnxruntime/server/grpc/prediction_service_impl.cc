#include "prediction_service_impl.h"
#include "../util.h"

namespace onnxruntime {
namespace server {
namespace grpc {

PredictionServiceImpl::PredictionServiceImpl(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env) : m_env(env) {}

::grpc::Status PredictionServiceImpl::Predict(::grpc::ServerContext* context, const ::onnxruntime::server::PredictRequest* request, ::onnxruntime::server::PredictResponse* response) {
  auto request_id = SetRequestContext(context);
  onnxruntime::server::Executor executor(m_env.get(), request_id);
  //TODO: (csteegz) Add modelspec for both paths.
  auto status = executor.Predict("default", "1", *request, *response);  // Currently only support one model so hard coded.
  if (!status.ok()) {
    //Based on reading https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/stubs/status.h the status codes map one to one.
    return ::grpc::Status(::grpc::StatusCode(status.error_code()), status.error_message());
  }
  return ::grpc::Status::OK;
}

std::string PredictionServiceImpl::SetRequestContext(::grpc::ServerContext* context) {
  auto metadata = context->client_metadata();
  auto search = metadata.find("x-ms-client-request-id");
  if (search != metadata.end()) {
    std::string id{search->second.data(), search->second.length()};
    context->AddInitialMetadata("x-ms-client-request-id", id);
  }
  auto request_id = internal::InternalRequestId();
  context->AddInitialMetadata("x-request-id", request_id);
  return request_id;
}

}  // namespace grpc
}  // namespace server

}  // namespace onnxruntime