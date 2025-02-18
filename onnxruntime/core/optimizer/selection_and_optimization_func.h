// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
static const std::string kConstantFoldingDQ = "ConstantFoldingDQ";

struct ConstantFoldingDQFuncs {
  static std::vector<std::unique_ptr<ComputeCapability>> Select(const GraphViewer& graph_viewer);
  static Status Optimize(Graph& graph, const ComputeCapability& optimization_cc, ComputeCapability& cc_to_update);
};
}  // namespace onnxruntime
