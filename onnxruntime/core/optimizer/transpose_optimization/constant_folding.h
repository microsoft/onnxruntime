// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "optimizer_api.h"

namespace onnx_transpose_optimization {
struct OptimizerCtx;

using CanModifyNodeFn = bool (*)(const OptimizerCtx& ctx, const api::NodeRef& node);

/// <summary>
/// Runs constant-folding on the graph managed by the provided OptimizerCtx. Only folds Transpose and Squeeze nodes
/// in order to remove Transpose/Squeeze nodes that were originally inserted to undo in-place modifications to
/// shared initializers.
/// </summary>
/// <param name="ctx">Optimizer context containing the graph</param>
/// <param name="can_modify_node_fn">Function that returns true if a node can be modified</param>
/// <returns>True if the graph was modified</returns>
bool RunConstantFolding(OptimizerCtx& ctx, CanModifyNodeFn can_modify_node_fn);
}  // namespace onnx_transpose_optimization
