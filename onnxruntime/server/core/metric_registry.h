// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <prometheus/registry.h>

namespace onnxruntime {
namespace server {
class MetricRegistry {
 private:
  static MetricRegistry* instance;

  MetricRegistry();
  ~MetricRegistry() = default;

public:
  MetricRegistry(const MetricRegistry&) = delete;
  MetricRegistry& operator=(const MetricRegistry&) = delete;

  // returns the global metric registry instance
  // use this handle to get the instance to increment/count requests
  static MetricRegistry* GetInstance();

  // Store quantile buckets 
  static prometheus::Histogram::BucketBoundaries TimeBuckets ();
  static prometheus::Histogram::BucketBoundaries ByteBuckets();

  // The internal prometheus registry
  prometheus::Registry registry;

  // The three metrics we're currently recording
  // Inference Time (model and version can be handled by labels so you can)
  // aggregate total onnxruntime server performance if desired
  prometheus::Family<prometheus::Histogram>* inferenceTimer;
  // Run Time (model and version again handled by labels)
  prometheus::Family<prometheus::Histogram>* runTimer;
  // Total number of HTTP requests split by path, this includes pinging the metric
  // endpoint and the health checker
  prometheus::Family<prometheus::Counter>* totalHTTPRequests;
  // Total number of gRPC requests received by the server
  prometheus::Family<prometheus::Counter>* totalGRPCRequests;
  // Total number of erroneous HTTP requests, includes error code and
  // for more in-depth analysis e.g bad inputs, protobuffer errors
  prometheus::Family<prometheus::Counter>* totalHTTPErrors;
  // Total number of erroneous gRPC requests, including error code
  prometheus::Family<prometheus::Counter>* totalGRPCErrors;
  // Request Sizes of HTTP Requests
  prometheus::Family<prometheus::Histogram>* httpRequestSize;
  // Request Sizes of gRPC Requests
  prometheus::Family<prometheus::Histogram>* grpcRequestSize;
};
}
}