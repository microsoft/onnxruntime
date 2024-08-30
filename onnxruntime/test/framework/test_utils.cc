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

static void CountOpsInGraphImpl(const Graph& graph, bool recurse_into_subgraphs, OpCountMap& ops) {
  for (auto& node : graph.Nodes()) {
    std::string key = node.Domain() + (node.Domain().empty() ? "" : ".") + node.OpType();

    ++ops[key];

    if (recurse_into_subgraphs && node.ContainsSubgraph()) {
      for (auto& subgraph : node.GetSubgraphs()) {
        CountOpsInGraphImpl(*subgraph, recurse_into_subgraphs, ops);
      }
    }
  }
}

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
OpCountMap CountOpsInGraph(const Graph& graph, bool recurse_into_subgraphs) {
  OpCountMap ops;
  CountOpsInGraphImpl(graph, recurse_into_subgraphs, ops);

  return ops;
}

}  // namespace test
}  // namespace onnxruntime
