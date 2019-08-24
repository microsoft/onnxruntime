// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_utils.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace test {
IExecutionProvider* TestCPUExecutionProvider() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return &cpu_provider;
}

#ifdef USE_CUDA
IExecutionProvider* TestCudaExecutionProvider() {
  static CUDAExecutionProviderInfo info;
  static CUDAExecutionProvider cuda_provider(info);
  return &cuda_provider;
}
#endif

#ifdef USE_TENSORRT
IExecutionProvider* TestTensorrtExecutionProvider() {
  static TensorrtExecutionProviderInfo info;
  static TensorrtExecutionProvider trt_provider(info);
  return &trt_provider;
}
#endif

#ifdef USE_OPENVINO
IExecutionProvider* TestOpenVINOExecutionProvider() {
  static OpenVINOExecutionProviderInfo info;
  static OpenVINOExecutionProvider openvino_provider(info);
  return &openvino_provider;
}
#endif

#ifdef USE_NNAPI
IExecutionProvider* TestNnapiExecutionProvider() {
  static NnapiExecutionProvider nnapi_provider;
  return &nnapi_provider;
}
#endif

static void CountOpsInGraphImpl(const Graph& graph, std::map<std::string, int>& ops) {
  for (auto& node : graph.Nodes()) {
    auto pos = ops.find(node.OpType());
    if (pos == ops.end()) {
      ops[node.OpType()] = 1;
    } else {
      ++pos->second;
    }

    if (node.ContainsSubgraph()) {
      for (auto& subgraph : node.GetSubgraphs()) {
        CountOpsInGraphImpl(*subgraph, ops);
      }
    }
  }
}

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
std::map<std::string, int> CountOpsInGraph(const Graph& graph) {
  std::map<std::string, int> ops;
  CountOpsInGraphImpl(graph, ops);

  return ops;
}

}  // namespace test
}  // namespace onnxruntime
