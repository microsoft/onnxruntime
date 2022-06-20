// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>
#include <unordered_map>

namespace onnxruntime {
class GraphViewer;
class Node;
class NodeUnit;

namespace xnnpack {
using PNode_PNodeUnit_Map = std::unordered_map<const Node*, const NodeUnit*>;
class NodeSupportChecker {
 public:
  NodeSupportChecker(const GraphViewer& graph,
                     const PNode_PNodeUnit_Map& supported_node_unit_map)
      : graph_{graph},
        supported_node_unit_map_{supported_node_unit_map} {
  }

  bool IsNodeSupported(const NodeUnit& node_unit);
  const NodeUnit* IsNodeSupportedWithFusion(const NodeUnit& node_unit);

 private:
  const GraphViewer& graph_;

  // previously selected nodes as of each time IsNodeSupport{WithFusion} is called.
  // updated in the background by the EP when it decides it will take a node_unit.
  const PNode_PNodeUnit_Map& supported_node_unit_map_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
