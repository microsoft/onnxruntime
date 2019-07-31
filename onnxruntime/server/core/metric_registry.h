#pragma once

#include <prometheus/registry.h>

namespace onnxruntime {
namespace server {
struct MetricRegistry {
  // returns the global metric registry instance
  // use this handle to get the instance to increment/count requests
  static MetricRegistry& Get() {
      static MetricRegistry metricRegistry;
      return metricRegistry;
  }

  // Store quantile buckets 
  static prometheus::Histogram::BucketBoundaries TimeBuckets () {
    // Anything greater than 10s we should bucket into +inf
    return {0, 50, 100, 250, 500, 1000, 1000, 2500, 5000, 10000};
  }

  static prometheus::Histogram::BucketBoundaries ByteBuckets() {
    // get megabyte boundaries?
    return {
      0, 500000, 1000000, 2000000, 3000000, 4000000,
      5000000, 6000000, 7000000, 8000000, 9000000, 10000000
    };
  }

  // The internal prometheus registry
  prometheus::Registry registry;
  // The three metrics we're currently recording
  // Inference Time (model and version can be handled by labels so you can)
  // aggregate total onnxruntime server performance if desired
  std::shared_ptr<prometheus::Family<prometheus::Histogram>*> inferenceTimer;
  // Total number of HTTP requests split by path, this includes pinging the metric
  // endpoint and the health checker
  std::shared_ptr<prometheus::Family<prometheus::Counter>*> totalHTTPRequests;
  // Total number of gRPC requests received by the server
  std::shared_ptr<prometheus::Family<prometheus::Counter>*> totalGRPCRequests;
  // Total number of erroneous HTTP requests, includes error code and
  // for more in-depth analysis e.g bad inputs, protobuffer errors
  std::shared_ptr<prometheus::Family<prometheus::Counter>*> totalHTTPErrors;
  // Total number of erroneous gRPC requests, including error code
  std::shared_ptr<prometheus::Family<prometheus::Counter>*> totalGRPCErrors;
  // Request Sizes of HTTP Requests
  std::shared_ptr<prometheus::Family<prometheus::Histogram>*> httpRequestSize;
  // Request Sizes of gRPC Requests
  std::shared_ptr<prometheus::Family<prometheus::Histogram>*> grpcRequestSize;

private:
  MetricRegistry() {
    // As this should be the exact same execution (executor.Predict()) for
    // both grpc and http and that we split model on label as a dimension
    // record this as a single metric
    // This is purely runtime/model performance so json/protobuf serde
    // shouldn't be considered, actual API Server performance can be done
    // through network tracing
    inferenceTimer = std::make_shared<prometheus::Family<prometheus::Histogram>*>(
      &prometheus::BuildHistogram().
        Name("ortserver_model_inference_time_milliseconds").
        Help("How long it took to perform inference").
        Register(registry)
    );
    
    totalHTTPRequests = std::make_shared<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_http_requests_total").
        Help("How many requests over HTTP the onnxruntime server has received").
        Register(registry)
    );

    totalGRPCRequests = std::make_shared<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_grpc_requests_total").
        Help("How many requests over gRPC the onnxruntime server has received").
        Register(registry)
    );

    totalHTTPErrors = std::make_shared<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_http_errors_total").
        Help("How many bad requests or errors the server has handled over HTTP").
        Register(registry)
    );
    
    totalGRPCErrors = std::make_shared<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_grpc_errors_total").
        Help("How many bad requests or errors the server has handled over gRPC").
        Register(registry)
    );

    httpRequestSize = std::make_shared<prometheus::Family<prometheus::Histogram>*>(
      &prometheus::BuildHistogram().
        Name("ortserver_http_request_size_bytes").
        Help("File sizes of http requests in bytes").
        Register(registry)
    );

    grpcRequestSize = std::make_shared<prometheus::Family<prometheus::Histogram>*>(
      &prometheus::BuildHistogram().
        Name("ortserver_grpc_request_size_bytes").
        Help("File sizes of grpc requests in bytes").
        Register(registry)
    );

  }


  ~MetricRegistry() = default;
};
}
}