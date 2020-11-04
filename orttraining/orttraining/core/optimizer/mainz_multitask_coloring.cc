// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/mainz_multitask_coloring.h"
#include "core/graph/graph_utils.h"
#include <set>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

const Node* MainzMultitaskColoring::SatisfyCondition(const Node& node) const {
  if (node.OpType() == "SoftmaxCrossEntropyLoss") {
    const auto next_node = node.OutputNodesBegin();
    if (next_node != node.OutputNodesEnd() && next_node->OpType() == "Div") {
      return next_node.operator->();
    }
  } else if (node.OpType() == "ReduceSum") {
    const auto next_node = node.OutputNodesBegin();
    if (next_node != node.OutputNodesEnd() && next_node->OpType() == "Add") {
      const auto next_next_node = next_node->OutputNodesBegin();
      if (next_next_node != next_node->OutputNodesEnd() && next_next_node->OpType() == "Div") {
        return next_next_node.operator->();
      }
    }
  }
  return nullptr;
}

Status MainzMultitaskColoring::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  std::vector<const Node*> start_nodes;
  for (auto node_index : node_topology_list) {
    Node& node = *graph.GetNode(node_index);
    const Node* start_node = SatisfyCondition(node);
    if (start_node) {
      start_nodes.push_back(start_node);
    }
  }

  if (start_nodes.size() < 3) {
    modified = false;
    return Status::OK();
  }
  
  std::vector<std::unordered_set<NodeIndex>> reachable_nodes(start_nodes.size());
  for (size_t i = 0; i < start_nodes.size(); ++i) {
    std::unordered_set<NodeIndex>& subgraph = reachable_nodes[i];
    // Reversely traverse to get reachable nodes.
    graph.ReverseDFSFrom(
        {start_nodes[i]}, {}, [&subgraph](const Node* n) { subgraph.insert(n->Index()); });
  }

  // verify subgraphs doesn't have intersection
  for (size_t i = 0; i < reachable_nodes.size(); ++i) {
    std::unordered_set<NodeIndex>& subgraph1 = reachable_nodes[i];
    for (size_t j = i + 1; j < reachable_nodes.size(); ++j) {
      std::unordered_set<NodeIndex>& subgraph2 = reachable_nodes[j];
      for (NodeIndex nid : subgraph1) {
        if (subgraph2.count(nid) != 0) {
          Node* node = graph.GetNode(nid);
          std::cout << node->Name() << " exists in subgraph " << i << " and " << j << "\n";
        }
      }
    }
  }

  for (size_t i = 0; i < reachable_nodes.size(); ++i) {
    for (NodeIndex nid : reachable_nodes[i]) {
      Node* node = graph.GetNode(nid);
      node->SetPriority(i);
    }
    std::cout << "Number of nodes in subgraph " << i << ": " << reachable_nodes[i].size() << "\n";
  }

  std::cout << "Total number of nodes: " << node_topology_list.size() << "\n";

  modified = true;
  return Status::OK();
}
}  // namespace onnxruntime
