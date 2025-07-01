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

#define RETURN_IF_API_ERROR(fn) \
  do {                          \
    Ort::Status status(fn);     \
    if (!status.IsOK()) {       \
      return status;            \
    }                           \
  } while (0)

namespace onnxruntime {
namespace test {
template <typename T>
struct VisitorPriorityQueue {
  using ComparatorType = std::function<bool(T, T)>;
  std::list<T> list_;
  const ComparatorType comparator_ = nullptr;
  VisitorPriorityQueue(const ComparatorType& comp) : comparator_(comp) {}

  void push(T node) {
    list_.insert(
        std::upper_bound(list_.begin(), list_.end(), node, comparator_),
        node);
  }
  bool empty() { return list_.empty(); }
  T top() { return list_.back(); }
  void pop() { list_.pop_back(); }
};

// Get the number of input edges that come from another node upstream.
static Ort::Status GetNodeInputEdgeCount(const OrtNode* node, size_t& num_input_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  OrtArrayOfConstObjects* inputs = nullptr;
  DeferOrtRelease<OrtArrayOfConstObjects> release_inputs(&inputs, ort_api.ReleaseArrayOfConstObjects);
  RETURN_IF_API_ERROR(ort_api.Node_GetInputs(node, &inputs));

  size_t num_inputs = 0;
  RETURN_IF_API_ERROR(ort_api.ArrayOfConstObjects_GetSize(inputs, &num_inputs));

  // Sum the number of inputs with a producer node.
  num_input_edges = 0;

  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValueInfo* input = nullptr;
    RETURN_IF_API_ERROR(ort_api.ArrayOfConstObjects_GetElementAt(inputs, i, reinterpret_cast<const void**>(&input)));
    if (input == nullptr) continue;  // Skip missing optional input

    const OrtNode* producer_node = nullptr;
    RETURN_IF_API_ERROR(ort_api.ValueInfo_GetValueProducer(input, &producer_node, /*output_index*/ nullptr));
    num_input_edges += static_cast<size_t>(producer_node != nullptr);
  }

  return Ort::Status{nullptr};
}

// Get all output nodes that consume an output from the given node.
static Ort::Status GetOutputNodes(const OrtNode* node, std::vector<const OrtNode*>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  OrtArrayOfConstObjects* outputs = nullptr;
  DeferOrtRelease<OrtArrayOfConstObjects> release_outputs(&outputs, ort_api.ReleaseArrayOfConstObjects);
  RETURN_IF_API_ERROR(ort_api.Node_GetOutputs(node, &outputs));

  size_t num_outputs = 0;
  RETURN_IF_API_ERROR(ort_api.ArrayOfConstObjects_GetSize(outputs, &num_outputs));

  std::vector<const OrtNode*> output_nodes;
  output_nodes.reserve(num_outputs);  // May have more than `num_outputs`

  // Gather the OrtNode consumers of every output.
  for (size_t i = 0; i < num_outputs; ++i) {
    const OrtValueInfo* output = nullptr;
    RETURN_IF_API_ERROR(ort_api.ArrayOfConstObjects_GetElementAt(outputs, i, reinterpret_cast<const void**>(&output)));
    if (output == nullptr) continue;  // Skip missing optional output

    size_t num_consumers = 0;
    RETURN_IF_API_ERROR(ort_api.ValueInfo_GetValueNumConsumers(output, &num_consumers));

    std::vector<const OrtNode*> node_consumers(num_consumers, nullptr);
    std::vector<int64_t> input_indices(num_consumers, 0);
    RETURN_IF_API_ERROR(ort_api.ValueInfo_GetValueConsumers(output, node_consumers.data(),
                                                            input_indices.data(), num_consumers));

    for (const OrtNode* consumer : node_consumers) {
      output_nodes.push_back(consumer);
    }
  }

  result = std::move(output_nodes);
  return Ort::Status{nullptr};
}

// Kahn's topological sort.
// Adapted from onnxruntime/core/graph/graph.cc to use public C API graph types.
static Ort::Status KahnsTopologicalSort(const OrtGraph& graph,
                                        const std::function<void(const OrtNode*)>& enter,
                                        const std::function<bool(const OrtNode*, const OrtNode*)>& comp) {
  const OrtApi& ort_api = Ort::GetApi();

  // Get all nodes
  size_t num_nodes = 0;
  RETURN_IF_API_ERROR(ort_api.Graph_GetNumNodes(&graph, &num_nodes));

  if (num_nodes == 0) {
    return Ort::Status{nullptr};  // Nothing to sort.
  }

  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_API_ERROR(ort_api.Graph_GetNodes(&graph, nodes.data(), nodes.size()));

  // Get the maximum node ID. Not really required if we chose to represent the `in_degree` as a map instead of vector.
  size_t max_node_id = 0;
  for (const OrtNode* node : nodes) {
    size_t node_id = 0;
    RETURN_IF_API_ERROR(ort_api.Node_GetId(node, &node_id));
    max_node_id = std::max(max_node_id, node_id);
  }

  std::vector<size_t> in_degree(max_node_id + 1, 0);
  std::vector<size_t> topo_order;
  VisitorPriorityQueue<const OrtNode*> to_visit(comp);

  topo_order.reserve(num_nodes);

  // Initialize in_degree and initial nodes to visit first.
  for (const OrtNode* node : nodes) {
    size_t input_edge_count = 0;
    RETURN_IF_API_ERROR(GetNodeInputEdgeCount(node, input_edge_count));

    size_t node_id = 0;
    RETURN_IF_API_ERROR(ort_api.Node_GetId(node, &node_id));

    in_degree[node_id] = input_edge_count;
    if (input_edge_count == 0) {
      to_visit.push(node);
    }
  }

  while (!to_visit.empty()) {
    const OrtNode* current_node = to_visit.top();
    to_visit.pop();

    if (!current_node) continue;

    if (enter) {
      enter(current_node);
    }

    std::vector<const OrtNode*> output_nodes;
    GetOutputNodes(current_node, output_nodes);

    for (const OrtNode* output_node : output_nodes) {
      size_t output_node_id = 0;
      RETURN_IF_API_ERROR(ort_api.Node_GetId(output_node, &output_node_id));

      auto& node_in_degree = in_degree[output_node_id];
      node_in_degree--;

      if (node_in_degree == 0) {
        to_visit.push(output_node);
      }
    }

    size_t current_node_id = 0;
    RETURN_IF_API_ERROR(ort_api.Node_GetId(current_node, &current_node_id));
    topo_order.push_back(current_node_id);
  }

  if (num_nodes != topo_order.size()) {
    return Ort::Status("Some nodes are not included in the topological sort: graph has a cycle", ORT_FAIL);
  }

  return Ort::Status{nullptr};
}

// Node comparison functor copied from onnxruntime/core/graph/graph.cc
struct PriorityNodeCompare {
  inline bool IsHighPri(const OrtNode* n) const {
    // local statics so we can compare std::strings in the checks
    static constexpr std::string_view shape_op("Shape");
    static constexpr std::string_view size_op("Size");

    const char* op_type = nullptr;
    Ort::Status status(Ort::GetApi().Node_GetOperatorType(n, &op_type));
    ORT_ENFORCE(status.IsOK());

    return shape_op == op_type || size_op == op_type;
  }

  // Used for std::priority_queue
  // If return false, n1 will be output first
  // If return true, n2 will be output first
  bool operator()(const OrtNode* n1, const OrtNode* n2) const {
    // nodes in global high priority list will be output first
    const bool isN1HighPri = IsHighPri(n1);
    const bool isN2HighPri = IsHighPri(n2);
    if (isN1HighPri != isN2HighPri) {
      return isN2HighPri;
    }

    // nodes with lower priority value will be output first
    const auto n1_priority = 0;  // n1->Priority(); // Looks to always be 0 inside ORT?
    const auto n2_priority = 0;  // n2->Priority(); // Looks to always be 0 inside ORT?
    if (n1_priority != n2_priority) {
      return n1_priority > n2_priority;
    }

    // otherwise, nodes with lower index will be output first
    size_t n1_id = 0;
    Ort::Status status1(Ort::GetApi().Node_GetId(n1, &n1_id));
    ORT_ENFORCE(status1.IsOK());

    size_t n2_id = 0;
    Ort::Status status2(Ort::GetApi().Node_GetId(n2, &n2_id));
    ORT_ENFORCE(status2.IsOK());

    return n1_id > n2_id;
  }
};

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
