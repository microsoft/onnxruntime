// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

namespace onnxruntime {
class GraphViewer;
class Node;

namespace xnnpack {
class NodeSupportChecker {
 public:
  NodeSupportChecker(const GraphViewer& graph,
                     const std::unordered_set<const Node*>& supported_nodes)
      : graph_{graph},
        supported_nodes_{supported_nodes} {
  }

  bool IsNodeSupported(const Node& node);
  const Node* IsNodeSupportedWithFusion(const Node& node);

 private:
  const GraphViewer& graph_;

  // previously selected nodes as of each time IsNodeSupport{WithFusion} is called.
  // updated in the background by the EP when it decides it will take a node.
  const std::unordered_set<const Node*>& supported_nodes_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
