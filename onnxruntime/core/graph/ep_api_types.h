// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/common/inlined_containers.h"
#include "core/graph/basic_types.h"
#include "core/graph/abi_graph_types.h"

namespace onnxruntime {
class Node;
class GraphViewer;

struct EpNode : public OrtNode {
  EpNode(const Node& node) : OrtNode(OrtNode::Type::kEditorNode), node(node) {}
  OrtNode* ToExternal() { return static_cast<OrtNode*>(this); }
  const OrtNode* ToExternal() const { return static_cast<const OrtNode*>(this); }

  const Node& node;
};

struct EpGraph : public OrtGraph {
  explicit EpGraph(const GraphViewer& g_viewer);
  OrtGraph* ToExternal() { return static_cast<OrtGraph*>(this); }
  const OrtGraph* ToExternal() const { return static_cast<const OrtGraph*>(this); }

  const GraphViewer& graph_viewer;
  InlinedVector<std::unique_ptr<EpNode>> nodes;
  InlinedHashMap<NodeIndex, EpNode*> index_to_node;
};
}  // namespace onnxruntime
