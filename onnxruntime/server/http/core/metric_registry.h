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
  // Total number of requests split by path, this includes pinging the metric
  // endpoint and the health checker
  prometheus::Family<prometheus::Counter>* totalRequests = nullptr;
  // Total number of erronious requests, includes error code and potentially message
  // for more in-depth analysis e.g bad inputs, protobuffer errors
  prometheus::Family<prometheus::Counter>* totalErrors = nullptr;

private:
  MetricRegistry() {
    inferenceTimer = &prometheus::BuildHistogram().
      Name("onnx_runtime_model_inference_time").
      Help("How long it took to perform inference").
      Register(registry);
    
    totalRequests = &prometheus::BuildCounter().
      Name("onnx_runtime_total_http_requests").
      Help("How many requests the onnxruntime server has received").
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