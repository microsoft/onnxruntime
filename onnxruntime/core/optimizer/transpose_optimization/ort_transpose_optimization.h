// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/transpose_optimization/optimizer_api.h"
#include "core/optimizer/transpose_optimization/onnx_transpose_optimization.h"

namespace onnxruntime {
/// <summary>
/// Get the extended handlers for ORT specific transpose optimization.
/// These include handlers for contrib ops, and where we have an NHWC version of a layout sensitive op.
/// </summary>
/// <returns>HandlerMap</returns>
const onnx_transpose_optimization::HandlerMap& OrtExtendedHandlers();

/// <summary>
/// Cost check function for transpose optimizer that takes into account implementation details of the
/// ORT execution provider kernels.
/// </summary>
/// <param name="graph">The graph being optimized</param>
/// <param name="node">The node we're considering pushing a Transpose through</param>
/// <param name="perm">The perm value of the Transpose</param>
/// <param name="outputs_leading_to_transpose">The set of outputs that lead to another Transpose in the graph.
///   If we can successfully push the Transpose until it meets another Transpose they can either cancel each other out,
///   or be merged into a single Transpose.
/// </param>
/// <returns>CostCheckResult indicating the action the transpose optimizer should perform.</returns>
onnx_transpose_optimization::CostCheckResult OrtEPCostCheck(
    const onnx_transpose_optimization::api::GraphRef& graph,
    const onnx_transpose_optimization::api::NodeRef& node,
    const std::vector<int64_t>& perm,
    const std::unordered_set<std::string>& outputs_leading_to_transpose);

/// <summary>
/// Swaps out a node for a new copy of that node with the specified op type and domain.
/// Current API does not allow nodes to have their op types or domains changed, so a new node is needed. All
/// attributes, inputs, and outputs are moved to the new node. The old node is removed from the graph and should no
/// longer be accessed.
/// </summary>
/// <param name="graph">Graph containing the node</param>
/// <param name="node">Node to copy and remove</param>
/// <param name="op_type">New node op_type</param>
/// <param name="domain">New node domain. "" for the default domain.</param>
/// <returns>The newly created node.</returns>
std::unique_ptr<onnx_transpose_optimization::api::NodeRef>
SwapNodeOpTypeAndDomain(onnx_transpose_optimization::api::GraphRef& graph,
                        onnx_transpose_optimization::api::NodeRef& node,
                        std::string_view op_type, std::string_view domain);

/// <summary>
/// Swaps out a node for a new copy of that node with the specified op type, domain, and since version.
/// Current API does not allow nodes to have their op types or domains changed, so a new node is needed. All
/// attributes, inputs, and outputs are moved to the new node. The old node is removed from the graph and should no
/// longer be accessed.
/// </summary>
/// <param name="graph">Graph containing the node</param>
/// <param name="node">Node to copy and remove</param>
/// <param name="op_type">New node op_type</param>
/// <param name="domain">New node domain. "" for the default domain.</param>
/// <param name="op_type">New node since version.</param>
/// <returns>The newly created node.</returns>
std::unique_ptr<onnx_transpose_optimization::api::NodeRef>
SwapNodeOpTypeDomainAndSinceVersion(onnx_transpose_optimization::api::GraphRef& graph,
                                    onnx_transpose_optimization::api::NodeRef& node,
                                    std::string_view op_type, std::string_view domain, int since_version);
}  // namespace onnxruntime
