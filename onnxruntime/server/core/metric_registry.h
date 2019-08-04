#pragma once

#include <prometheus/registry.h>

namespace onnxruntime {
namespace server {
class MetricRegistry {
public:
  // returns the global metric registry instance
  // use this handle to get the instance to increment/count requests
  static MetricRegistry& Get() {
      static MetricRegistry metricRegistry;
      return metricRegistry;
  }

  // Store quantile buckets 
  static prometheus::Histogram::BucketBoundaries TimeBuckets () {
    // Anything greater than 10s we should bucket into +inf
    return {0, 50, 100, 250, 500, 1000, 2500, 5000, 10000};
  }

  static prometheus::Histogram::BucketBoundaries ByteBuckets() {
    // get megabyte boundaries?
    return {
      // 0, 0.5, 1, 2, 5, 10, 20 MiB
      0, 524288, 1048576, 2097152, 5242880, 10485760, 20971520
    };
  }

  // The internal prometheus registry
  prometheus::Registry registry;
  // The three metrics we're currently recording
  // Inference Time (model and version can be handled by labels so you can)
  // aggregate total onnxruntime server performance if desired
  std::unique_ptr<prometheus::Family<prometheus::Histogram>*> inferenceTimer;
  // Run Time (model and version again handled by labels)
  std::unique_ptr<prometheus::Family<prometheus::Histogram>*> runTimer;
  // Total number of HTTP requests split by path, this includes pinging the metric
  // endpoint and the health checker
  std::unique_ptr<prometheus::Family<prometheus::Counter>*> totalHTTPRequests;
  // Total number of gRPC requests received by the server
  std::unique_ptr<prometheus::Family<prometheus::Counter>*> totalGRPCRequests;
  // Total number of erroneous HTTP requests, includes error code and
  // for more in-depth analysis e.g bad inputs, protobuffer errors
  std::unique_ptr<prometheus::Family<prometheus::Counter>*> totalHTTPErrors;
  // Total number of erroneous gRPC requests, including error code
  std::unique_ptr<prometheus::Family<prometheus::Counter>*> totalGRPCErrors;
  // Request Sizes of HTTP Requests
  std::unique_ptr<prometheus::Family<prometheus::Histogram>*> httpRequestSize;
  // Request Sizes of gRPC Requests
  std::unique_ptr<prometheus::Family<prometheus::Histogram>*> grpcRequestSize;

private:
  MetricRegistry() {
    // As this should be the exact same execution (executor.Predict()) for
    // both grpc and http and that we split model on label as a dimension
    // record this as a single metric
    // This is purely runtime/model performance so json/protobuf serde
    // shouldn't be considered, actual API Server performance can be done
    // through network tracing
    inferenceTimer = std::make_unique<prometheus::Family<prometheus::Histogram>*>(
      &prometheus::BuildHistogram().
        Name("ortserver_model_inference_time_milliseconds").
        Help("How long it took to perform inference").
        Register(registry)
    );

    runTimer = std::make_unique<prometheus::Family<prometheus::Histogram>*>(
      &prometheus::BuildHistogram().
        Name("ortserver_model_session_run_time_milliseconds").
        Help("How long it took to perform Session.Run on the input, includes casting").
        Register(registry)
    );
    
    totalHTTPRequests = std::make_unique<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_http_requests_total").
        Help("How many requests over HTTP the onnxruntime server has received").
        Register(registry)
    );

    totalGRPCRequests = std::make_unique<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_grpc_requests_total").
        Help("How many requests over gRPC the onnxruntime server has received").
        Register(registry)
    );

    totalHTTPErrors = std::make_unique<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_http_errors_total").
        Help("How many bad requests or errors the server has handled over HTTP").
        Register(registry)
    );
    
    totalGRPCErrors = std::make_unique<prometheus::Family<prometheus::Counter>*>(
      &prometheus::BuildCounter().
        Name("ortserver_grpc_errors_total").
        Help("How many bad requests or errors the server has handled over gRPC").
        Register(registry)
    );

    httpRequestSize = std::make_unique<prometheus::Family<prometheus::Histogram>*>(
      &prometheus::BuildHistogram().
        Name("ortserver_http_request_size_bytes").
        Help("File sizes of http requests in bytes").
        Register(registry)
    );

    grpcRequestSize = std::make_unique<prometheus::Family<prometheus::Histogram>*>(
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