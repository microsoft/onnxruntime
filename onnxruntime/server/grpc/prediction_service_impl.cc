#include <chrono> // high res clock for inference monitoring

#include "prediction_service_impl.h"
#include "request_id.h"
#include "metric_registry.h"

namespace onnxruntime {
namespace server {
namespace grpc {

PredictionServiceImpl::PredictionServiceImpl(const std::shared_ptr<onnxruntime::server::ServerEnvironment>& env) : environment_(env) {}

::grpc::Status PredictionServiceImpl::Predict(::grpc::ServerContext* context, const ::onnxruntime::server::PredictRequest* request, ::onnxruntime::server::PredictResponse* response) {
  auto request_id = SetRequestContext(context);
  // No labels for gRPC counter as we don't have model arguments
  MetricRegistry::Get().totalGRPCRequests->Add({}).Increment();
  // Log Request size
  MetricRegistry::Get().grpcRequestSize->
    Add({}, MetricRegistry::ByteBuckets()).
    Observe(request->ByteSize());
  onnxruntime::server::Executor executor(environment_.get(), request_id);
  auto begin = std::chrono::high_resolution_clock::now();
  //TODO: (csteegz) Add modelspec for both paths.
  auto status = executor.Predict("default", "1", *request, *response);  // Currently only support one model so hard coded.
  auto end = std::chrono::high_resolution_clock::now();
  if (!status.ok()) {
    // Record error on prometheus
    MetricRegistry::Get().totalGRPCErrors->Add({
      {"errorCode", std::to_string(static_cast<unsigned>(status.error_code()))},
    }).Increment();
    return ::grpc::Status(::grpc::StatusCode(status.error_code()), status.error_message());
  }
  // See above, currently only support one model so hardcoded
  MetricRegistry::Get().inferenceTimer->Add({{"name", "default"}, {"version", "1"}},
      MetricRegistry::TimeBuckets()).
      // Casting for MS
      Observe(std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count());
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