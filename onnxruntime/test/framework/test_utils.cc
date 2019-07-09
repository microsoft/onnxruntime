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
  static TensorrtExecutionProvider trt_provider;
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

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
std::map<std::string, int> CountOpsInGraph(const Graph& graph) {
  std::map<std::string, int> op_to_count;
  for (auto& node : graph.Nodes()) {
    op_to_count[node.OpType()] =
        op_to_count.count(node.OpType()) == 0 ? 1 : ++op_to_count[node.OpType()];
  }
  return op_to_count;
}

}  // namespace test
}  // namespace onnxruntime
