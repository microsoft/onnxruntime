// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "metric_registry.h"

namespace onnxruntime {
namespace server {

MetricRegistry* MetricRegistry::instance = 0;

MetricRegistry::MetricRegistry() {
  // As this should be the exact same execution (executor.Predict()) for
  // both grpc and http and that we split model on label as a dimension
  // record this as a single metric
  // This is purely runtime/model performance so json/protobuf serde
  // shouldn't be considered, actual API Server performance can be done
  // through network tracing
  inferenceTimer = &prometheus::BuildHistogram().
      Name("ortserver_model_inference_time_milliseconds").
      Help("How long it took to perform inference").
      Register(registry);

  runTimer = &prometheus::BuildHistogram().
      Name("ortserver_model_session_run_time_milliseconds").
      Help("How long it took to perform Session.Run on the input, includes casting").
      Register(registry);

  totalHTTPRequests = &prometheus::BuildCounter().
      Name("ortserver_http_requests_total").
      Help("How many requests over HTTP the onnxruntime server has received").
      Register(registry);

  totalGRPCRequests = &prometheus::BuildCounter().
      Name("ortserver_grpc_requests_total").
      Help("How many requests over gRPC the onnxruntime server has received").
      Register(registry);

  totalHTTPErrors = &prometheus::BuildCounter().
      Name("ortserver_http_errors_total").
      Help("How many bad requests or errors the server has handled over HTTP").
      Register(registry);

  totalGRPCErrors = &prometheus::BuildCounter().
      Name("ortserver_grpc_errors_total").
      Help("How many bad requests or errors the server has handled over gRPC").
      Register(registry);

  httpRequestSize = &prometheus::BuildHistogram().
      Name("ortserver_http_request_size_bytes").
      Help("File sizes of http requests in bytes").
      Register(registry);

  grpcRequestSize = &prometheus::BuildHistogram().
      Name("ortserver_grpc_request_size_bytes").
      Help("File sizes of grpc requests in bytes").
      Register(registry);
}

MetricRegistry* MetricRegistry::GetInstance() {
  if (instance == nullptr) {
    instance = new MetricRegistry();
  }

  return instance;
}

prometheus::Histogram::BucketBoundaries MetricRegistry::TimeBuckets() {
  // Anything greater than 10s we should bucket into +inf
  return {0, 50, 100, 250, 500, 1000, 2500, 5000, 10000};
}

prometheus::Histogram::BucketBoundaries MetricRegistry::ByteBuckets() {
  // get megabyte boundaries?
  return {
      // 0, 0.5, 1, 2, 5, 10, 20 MiB
      0, 524288, 1048576, 2097152, 5242880, 10485760, 20971520
  };
}

}
}
