// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "core/common/common.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/test_environment.h"

#define RETURN_IF_API_ERROR(fn) \
  do {                          \
    Ort::Status status(fn);     \
    if (!status.IsOK()) {       \
      return status;            \
    }                           \
  } while (0)

struct OrtGraph;
namespace onnxruntime {
namespace test {

/// <summary>
/// Utility that loads a model from file and provides a OrtGraph view of the model for testing the public graph APIs.
/// </summary>
class TestGraph {
 public:
  explicit TestGraph(std::shared_ptr<Model> model);
  ~TestGraph();

  static std::unique_ptr<TestGraph> Load(const ORTCHAR_T* model_path);
  const OrtGraph& GetOrtGraph() const;
  const GraphViewer& GetGraphViewer() const;
  const Model& GetModel() const;

 private:
  std::shared_ptr<Model> model;
  GraphViewer graph_viewer;
  std::unique_ptr<OrtGraph> api_graph;
};

struct NodeArgConsumer {
  NodeArgConsumer(const Node* node, int64_t index) : node(node), input_index(index) {}
  const Node* node = nullptr;
  int64_t input_index = -1;
};

// Helper to release Ort one or more objects obtained from the public C API at the end of their scope.
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }
  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

// Returns consumers (i.e., consumer node + input index) of a NodeArg from the original graph.
Status GetNodeArgConsumers(const GraphViewer& graph_viewer, const NodeArg& node_arg,
                           /*out*/ std::vector<NodeArgConsumer>& consumers);

// Get output index for the given NodeArg name. Returns error if the node does not produce that node arg as an output.
Status GetOutputIndex(const Node& producer_node, const std::string& name, /*out*/ size_t& index);

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

  size_t num_inputs = 0;
  RETURN_IF_API_ERROR(ort_api.Node_GetNumInputs(node, &num_inputs));

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_IF_API_ERROR(ort_api.Node_GetInputs(node, inputs.data(), inputs.size()));

  // Sum the number of inputs with a producer node.
  num_input_edges = 0;

  for (const OrtValueInfo* ort_input : inputs) {
    Ort::ConstValueInfo input{ort_input};
    if (input == nullptr) continue;  // Skip missing optional input

    auto producer_info = input.GetProducerNode();
    num_input_edges += static_cast<size_t>(producer_info.node != nullptr);
  }

  return Ort::Status{nullptr};
}

// Get all output nodes that consume an output from the given node.
static Ort::Status GetOutputNodes(const OrtNode* node, std::vector<Ort::ConstNode>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_outputs = 0;
  RETURN_IF_API_ERROR(ort_api.Node_GetNumOutputs(node, &num_outputs));

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_IF_API_ERROR(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()));

  std::vector<Ort::ConstNode> output_nodes;
  output_nodes.reserve(num_outputs);  // May have more than `num_outputs`

  // Gather the OrtNode consumers of every output.
  for (const OrtValueInfo* ort_output : outputs) {
    Ort::ConstValueInfo output{ort_output};
    if (output == nullptr) continue;  // Skip missing optional output

    auto consumers_info = output.GetConsumers();
    for (const auto& consumer : consumers_info) {
      output_nodes.push_back(consumer.node);
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

  try {
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

      std::vector<Ort::ConstNode> output_nodes;
      RETURN_IF_API_ERROR(GetOutputNodes(current_node, output_nodes));

      for (const auto& output_node : output_nodes) {
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
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status;
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status;
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

}  // namespace test
}  // namespace onnxruntime
