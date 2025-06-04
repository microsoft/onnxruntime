// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <string>
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
  // A grouping of supported nodes that are executed by a specific hardware device.
  struct NodeGrouping {
    const OrtHardwareDevice* hardware_device;  // The hw device that executes the supported nodes.
    onnxruntime::InlinedVector<const onnxruntime::EpNode*> nodes;
  };

  explicit OrtEpGraphSupportInfo(const onnxruntime::EpGraph& graph) : ort_graph(graph) {}
  onnxruntime::Status AddSupportedNodes(const OrtHardwareDevice* hardware_device,
                                        gsl::span<const OrtNode* const> nodes);

  const onnxruntime::EpGraph& ort_graph;
  onnxruntime::InlinedVector<NodeGrouping> node_groupings;
};
