// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
static const std::string kCONSTANT_FOLDING_DQ = "ConstantFoldingDQ";

// ConstantFoldingDQ selection function
std::vector<std::unique_ptr<ComputeCapability>> ConstantFoldingDQ_selection(const GraphViewer& graph_viewer);

// ConstantFoldingDQ optimization function
Status ConstantFoldingDQ_optimization(Graph& graph, const ComputeCapability& optimization_cc, ComputeCapability& cc_to_update);



}  // namespace onnxruntime
