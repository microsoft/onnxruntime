// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <string>
#include <utility>
#include <vector>

#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct EpGraph;
struct EpNode;
}  // namespace onnxruntime

/// <summary>
/// Class used specify the nodes an EP supports. An instance of this class is passed to OrtEp's
/// GetCapability() function. An OrtEp adds groups of supported nodes to the OrtEpGraphSupportInfo instance.
/// </summary>
struct OrtEpGraphSupportInfo {
  enum class NodeGroupingKind {
    kInvalidGrouping = 0,
    kSingleAssignedNode,
    kFusedNode,
  };

  // A grouping of supported nodes that should be handled in a single ComputeCapability.
  struct NodeGrouping {
    NodeGrouping(NodeGroupingKind kind, std::vector<const onnxruntime::EpNode*>&& nodes,
                 const OrtNodeFusionOptions& fusion_options = {})
        : kind(kind), nodes(std::move(nodes)), fusion_options(fusion_options) {}

    NodeGroupingKind kind = NodeGroupingKind::kInvalidGrouping;
    std::vector<const onnxruntime::EpNode*> nodes;
    OrtNodeFusionOptions fusion_options = {};
  };

  explicit OrtEpGraphSupportInfo(const onnxruntime::EpGraph& graph) : ort_graph(graph) {}

  onnxruntime::Status AddNodesToFuse(gsl::span<const OrtNode* const> nodes,
                                     const OrtNodeFusionOptions* node_fusion_options = nullptr);
  onnxruntime::Status AddSingleNode(const OrtNode* node);

  const onnxruntime::EpGraph& ort_graph;
  std::vector<NodeGrouping> node_groupings;
};
