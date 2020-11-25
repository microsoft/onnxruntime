// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prediction_service_impl.h"
#include "request_id.h"

namespace onnxruntime {
namespace server {
namespace grpc {

PredictionServiceImpl::PredictionServiceImpl(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env) : environment_(env) {}

::grpc::Status PredictionServiceImpl::Predict(::grpc::ServerContext* context, const ::onnxruntime::server::PredictRequest* request, ::onnxruntime::server::PredictResponse* response) {
  auto request_id = SetRequestContext(context);
  onnxruntime::server::Executor executor(environment_.get(), request_id);
  //TODO: (csteegz) Add modelspec for both paths.
  auto status = executor.Predict("default", "1", *request, *response);  // Currently only support one model so hard coded.
  if (!status.ok()) {
    return ::grpc::Status(::grpc::StatusCode(status.error_code()), status.error_message());
  }
  return ::grpc::Status::OK;
}

std::string PredictionServiceImpl::SetRequestContext(::grpc::ServerContext* context) {
  auto metadata = context->client_metadata();
  auto request_id = util::InternalRequestId();
  context->AddInitialMetadata(util::MS_REQUEST_ID_HEADER, request_id);
  auto logger = environment_->GetLogger(request_id);
  auto search = metadata.find(util::MS_CLIENT_REQUEST_ID_HEADER);
  if (search != metadata.end()) {
    std::string id{search->second.data(), search->second.length()};
    context->AddInitialMetadata(util::MS_CLIENT_REQUEST_ID_HEADER, id);
    logger->info("{}: [{}]", util::MS_CLIENT_REQUEST_ID_HEADER, id);
  }

  return request_id;
}

}  // namespace grpc
}  // namespace server

}  // namespace onnxruntime