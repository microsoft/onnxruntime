// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/execution_provider.h"  // for IExecutionProvider::IKernelLookup
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace logging {
class Logger;
}

/**
  Returns a list of nodes that are preferred on CPU.
  They are commonly shape-related computation subgraphs.
  @param graph Graph viewer
  @param kernel_lookup The kernel lookup for the target execution provider
  @param tentative_nodes Nodes that are tentative to be placed on on target EP
  */
std::unordered_set<NodeIndex> GetCpuPreferredNodes(const GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes,
                                                   const logging::Logger& logger);

}  // namespace onnxruntime
