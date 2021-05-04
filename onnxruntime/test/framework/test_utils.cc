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

#ifdef USE_ROCM
IExecutionProvider* TestRocmExecutionProvider() {
  static ROCMExecutionProviderInfo info;
  static ROCMExecutionProvider rocm_provider(info);
  return &rocm_provider;
}
#endif

#ifdef USE_TENSORRT
#if 0  // TODO: TensorRT is shared, can't access these directly anymore
IExecutionProvider* TestTensorrtExecutionProvider() {
  static TensorrtExecutionProviderInfo info;
  static TensorrtExecutionProvider trt_provider(info);
  return &trt_provider;
}
#endif
#endif

#ifdef USE_OPENVINO
#if 0  // TODO: OpenVINO is shared, can't access these directly anymore
IExecutionProvider* TestOpenVINOExecutionProvider() {
  static OpenVINOExecutionProviderInfo info;
  static OpenVINOExecutionProvider openvino_provider(info);
  return &openvino_provider;
}
#endif
#endif

#ifdef USE_NNAPI
IExecutionProvider* TestNnapiExecutionProvider() {
  static NnapiExecutionProvider nnapi_provider(0);
  return &nnapi_provider;
}
#endif

#ifdef USE_RKNPU
IExecutionProvider* TestRknpuExecutionProvider() {
  static RknpuExecutionProvider rknpu_provider;
  return &rknpu_provider;
}
#endif

#ifdef USE_COREML
IExecutionProvider* TestCoreMLExecutionProvider(uint32_t coreml_flags) {
  static CoreMLExecutionProvider coreml_provider(coreml_flags);
  return &coreml_provider;
}
#endif

static void CountOpsInGraphImpl(const Graph& graph, bool recurse_into_subgraphs, std::map<std::string, int>& ops) {
  for (auto& node : graph.Nodes()) {
    std::string key = node.Domain() + (node.Domain().empty() ? "" : ".") + node.OpType();

    auto pos = ops.find(key);
    if (pos == ops.end()) {
      ops[key] = 1;
    } else {
      ++pos->second;
    }

    if (recurse_into_subgraphs && node.ContainsSubgraph()) {
      for (auto& subgraph : node.GetSubgraphs()) {
        CountOpsInGraphImpl(*subgraph, recurse_into_subgraphs, ops);
      }
    }
  }
}

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
std::map<std::string, int> CountOpsInGraph(const Graph& graph, bool recurse_into_subgraphs) {
  std::map<std::string, int> ops;
  CountOpsInGraphImpl(graph, recurse_into_subgraphs, ops);

  return ops;
}

}  // namespace test
}  // namespace onnxruntime
