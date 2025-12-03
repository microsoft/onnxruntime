// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/gsl>
#include <functional>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

#include "core/common/common.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/ep_graph/test_ep_graph_utils.h"

//
// Test implementation of Kahn's Topological sort using public C graph APIs and C++ STL.
//

namespace onnxruntime {
namespace test {

TEST(EpGraphTest, BasicKahnTopoSort) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/bart_tiny.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  // Sort OrtGraph with a custom Kahn's topological sorting algorithm.
  std::vector<size_t> api_nodes_topo_sort_with_priority;
  Ort::Status status(KahnsTopologicalSort(
      test_graph->GetOrtGraph(),
      [&](const OrtNode* node) {
        size_t node_id = 0;
        Ort::Status status(Ort::GetApi().Node_GetId(node, &node_id));
        ORT_ENFORCE(status.IsOK());

        api_nodes_topo_sort_with_priority.push_back(node_id);
      },
      PriorityNodeCompare()));
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Use ORT's built in sorting with priority.
  std::vector<size_t> ort_topo_sort_with_priority = test_graph->GetGraphViewer()
                                                        .GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);

  // Check that they are equal.
  ASSERT_EQ(api_nodes_topo_sort_with_priority, ort_topo_sort_with_priority);
}
}  // namespace test
}  // namespace onnxruntime
