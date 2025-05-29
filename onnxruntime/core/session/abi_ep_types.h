// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct EpGraph;
struct EpNode;
}  // namespace onnxruntime

struct OrtEpGraphSupportInfo {
  struct Subgraph {
    std::string name;
    const OrtHardwareDevice* hardware_device;
    std::vector<const onnxruntime::EpNode*> nodes;
  };

  explicit OrtEpGraphSupportInfo(const onnxruntime::EpGraph& graph) : ort_graph(graph) {}
  onnxruntime::Status AddSubgraph(const char* name, const OrtHardwareDevice* hardware_device, gsl::span<const OrtNode* const> nodes);

  const onnxruntime::EpGraph& ort_graph;
  std::vector<Subgraph> subgraphs;
};
