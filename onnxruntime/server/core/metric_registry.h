#pragma once

#include <prometheus/registry.h>

namespace onnxruntime {
namespace server {
struct MetricRegistry {
  // returns the global metric registry instance
  // use this handle to get the instance to increment/count requests
  static MetricRegistry& get() {
      static MetricRegistry metricRegistry;
      return metricRegistry;
  }

  // Store quantile buckets 
  static prometheus::Histogram::BucketBoundaries buckets () {
    // Anything greater than 10s we should bucket into +inf
    return {0, 50, 100, 250, 500, 1000, 1000, 2500, 5000, 10000};
  }

  // The internal prometheus registry
  prometheus::Registry registry;
  // The three metrics we're currently recording
  // Inference Time (model and version can be handled by labels so you can)
  // aggregate total onnxruntime server performance if desired
  prometheus::Family<prometheus::Histogram>* inferenceTimer = nullptr;
  // Total number of HTTP requests split by path, this includes pinging the metric
  // endpoint and the health checker
  prometheus::Family<prometheus::Counter>* totalHTTPRequests = nullptr;
  // Total number of gRPC requests received by the server
  prometheus::Family<prometheus::Counter>* totalGRPCRequests = nullptr;
  // Total number of erronious requests, includes error code and potentially message
  // for more in-depth analysis e.g bad inputs, protobuffer errors
  prometheus::Family<prometheus::Counter>* totalErrors = nullptr;

private:
  MetricRegistry() {
    // As this should be the exact same execution (executor.Predict()) for
    // both grpc and http and that we split model on label as a dimension
    // record this as a single metric
    // This is purely runtime/model performance so json/protobuf serde
    // shouldn't be considered, actual API Server performance can be done
    // through network tracing
    inferenceTimer = &prometheus::BuildHistogram().
      Name("onnx_runtime_model_inference_time").
      Help("How long it took to perform inference").
      Register(registry);
    
    totalHTTPRequests = &prometheus::BuildCounter().
      Name("onnx_runtime_total_http_requests").
      Help("How many requests over HTTP the onnxruntime server has received").
      Register(registry);

    totalGRPCRequests = &prometheus::BuildCounter().
      Name("onnx_runtime_total_grpc_requests").
      Help("How many requests over gRPC the onnxruntime server has received").
      Register(registry);

    totalErrors = &prometheus::BuildCounter().
      Name("onnx_runtime_total_errors").
      Help("How many bad requests or errors the server has handled").
      Register(registry);
  }


  ~MetricRegistry() = default;
};
}
}