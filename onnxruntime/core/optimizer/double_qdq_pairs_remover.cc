// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "double_qdq_pairs_remover.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {
Status
DoubleQDQPairsRemover::ApplyImpl(
    Graph &graph,
    bool &modified,
    int /*graph_level*/,
    const logging::Logger &logger
                                ) const {
  const GraphViewer graph_viewer(graph);
  const auto &node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (const auto &node_index: node_topology_list) {
    NodeIndex parent_index = 0;
    NodeIndex child_index = 0;
    NodeIndex grandchild_index = 0;
    if (IsNodeRemovable(graph, logger, node_index, parent_index, child_index, grandchild_index)) {
      graph.RemoveEdge(parent_index, node_index, 0, 0);
      graph.RemoveEdge(node_index, child_index, 0, 0);
      graph.RemoveEdge(child_index, grandchild_index, 0, 0);
      graph_utils::ReplaceNodeInput(*graph.GetNode(grandchild_index), 0, *graph.GetNode(node_index)
          ->MutableInputDefs()[0]);
      graph.AddEdge(parent_index, grandchild_index, 0, 0);
      graph.RemoveNode(child_index);
      graph.RemoveNode(node_index);
      modified = true;
    }
  }
  return Status::OK();
}

bool
DoubleQDQPairsRemover::IsNodeRemovable(
    const Graph &graph,
    const logging::Logger &logger,
    const NodeIndex &node_index,
    NodeIndex &parent_index,
    NodeIndex &child_index,
    NodeIndex &grandchild_index
                                      ) {
  // Check if the node is a DQ node
  const Node *node = graph.GetNode(node_index);
  if (node == nullptr || node->OpType() != "DequantizeLinear" || node->GetInputEdgesCount() != 1) { return false; }

  // parent should be a Q node, and have only one perent
  parent_index = node->InputEdgesBegin()->GetNode().Index();
  const Node *parent = graph.GetNode(parent_index);
  if (parent == nullptr || parent->OpType() != "QuantizeLinear") { return false; }

  // child should be a Q node, and have only one child
  child_index = node->OutputEdgesBegin()->GetNode().Index();
  const Node *child = graph.GetNode(child_index);
  if (child == nullptr || child->OpType() != "QuantizeLinear") { return false; }
  if (node->GetOutputEdgesCount() != 1) {
    LOGS(logger, WARNING)
      << "GraphTransformer:DoubleQDQPairsRemover: Found more than one Q node under DQ node " << node->Name()
      << ". Please run IdenticalChildrenConsolidation before DoubleQDQPairsRemover. ";
    return false;
  }

  // grandchild should be a DQ node, and have only one grandchild
  grandchild_index = child->OutputEdgesBegin()->GetNode().Index();
  const Node *grandchild = graph.GetNode(grandchild_index);
  if(grandchild == nullptr || grandchild->OpType() != "DequantizeLinear") {return false;}
  if (child->GetOutputEdgesCount() != 1) {
    LOGS(logger, WARNING)
      << "GraphTransformer:DoubleQDQPairsRemover: Found more than one DQ node under Q node " << child->Name()
      << ". Please run IdenticalChildrenConsolidation before DoubleQDQPairsRemover. ";
    return false;
  }
  return true;
}

}  // namespace onnxruntime
