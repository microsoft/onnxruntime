// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include <gsl/gsl>

namespace onnxruntime {

/**
  Returns a list of nodes that are preferred on CPU.
  They are commonly shape-related computation subgraphs.
  @param graph Graph viewer
  @param provider_type The target execution provider type
  @param kernel_registries Kernel registries for the target EP
  @param tentative_nodes Nodes that are tentative to be placed on on target EP
  */
  std::unordered_set<NodeIndex> GetCpuPreferredNodes(const GraphViewer& graph,
                                                    const std::string& provider_type,
                                                    gsl::span<const KernelRegistry* const> kernel_registries,
                                                    gsl::span<const NodeIndex> tentative_nodes);

}  // namespace onnxruntime
