// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/ep_api_types.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"

namespace onnxruntime {
EpGraph::EpGraph(const GraphViewer& g_viewer) : OrtGraph(OrtGraph::Type::kEpGraph), graph_viewer(g_viewer) {
  nodes.reserve(g_viewer.NumberOfNodes());
  for (const Node& node : g_viewer.Nodes()) {
    nodes.push_back(std::make_unique<EpNode>(node));
    index_to_node[node.Index()] = nodes.back().get();
  }
}

}  // namespace onnxruntime
