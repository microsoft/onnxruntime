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

/// <summary>
/// Concrete implementation of OrtNode used in the OrtEpApi.
/// </summary>
struct EpNode : public OrtNode {
  explicit EpNode(const Node& node) : OrtNode(OrtGraphIrApi::kEpApi), node(node) {}
  OrtNode* ToExternal() { return static_cast<OrtNode*>(this); }
  const OrtNode* ToExternal() const { return static_cast<const OrtNode*>(this); }

  static EpNode* ToInternal(OrtNode* ort_node) {
    return ort_node->graph_ir_api == OrtGraphIrApi::kEpApi ? static_cast<EpNode*>(ort_node) : nullptr;
  }

  static const EpNode* ToInternal(const OrtNode* ort_node) {
    return ort_node->graph_ir_api == OrtGraphIrApi::kEpApi ? static_cast<const EpNode*>(ort_node) : nullptr;
  }

  const Node& node;
};

/// <summary>
/// Concrete implementation of OrtGraph used in the OrtEpApi.
/// </summary>
struct EpGraph : public OrtGraph {
  explicit EpGraph(const GraphViewer& g_viewer);
  OrtGraph* ToExternal() { return static_cast<OrtGraph*>(this); }
  const OrtGraph* ToExternal() const { return static_cast<const OrtGraph*>(this); }

  static EpGraph* ToInternal(OrtGraph* ort_graph) {
    return ort_graph->graph_ir_api == OrtGraphIrApi::kEpApi ? static_cast<EpGraph*>(ort_graph) : nullptr;
  }
  static const EpGraph* ToInternal(const OrtGraph* ort_graph) {
    return ort_graph->graph_ir_api == OrtGraphIrApi::kEpApi ? static_cast<const EpGraph*>(ort_graph) : nullptr;
  }

  const GraphViewer& graph_viewer;
  InlinedVector<std::unique_ptr<EpNode>> nodes;
  InlinedHashMap<NodeIndex, EpNode*> index_to_node;
};
}  // namespace onnxruntime
