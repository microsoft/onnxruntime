// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct EpGraph;
struct EpNode;
}  // namespace onnxruntime

struct OrtNodeFusionOptions {
  std::vector<const onnxruntime::EpNode*> nodes;
  bool drop_constant_initializers = false;
};

/// <summary>
/// Class used specify the nodes an EP supports. An instance of this class is passed to OrtEp's
/// GetCapability() function. An OrtEp adds groups of supported nodes to the OrtEpGraphSupportInfo instance.
/// </summary>
struct OrtEpGraphSupportInfo {
  // A grouping of supported nodes that should be handled in a single ComputeCapability.
  struct NodeGrouping {
    NodeGrouping(const onnxruntime::EpNode* single_node) : variant_(single_node) {}
    NodeGrouping(const OrtNodeFusionOptions& node_fusion_options) : variant_(node_fusion_options) {}

    const OrtNodeFusionOptions* TryGetNodeFusionOptions() const;
    const onnxruntime::EpNode* TryGetSingleNode() const;

   private:
    std::variant<OrtNodeFusionOptions, const onnxruntime::EpNode*> variant_ = {};
  };

  explicit OrtEpGraphSupportInfo(const onnxruntime::EpGraph& graph) : ort_graph(graph) {}

  void AddNodesToFuse(const OrtNodeFusionOptions& node_fusion_options);
  void AddSingleNode(const onnxruntime::EpNode& node);

  const onnxruntime::EpGraph& ort_graph;
  std::vector<NodeGrouping> node_groupings;
};
