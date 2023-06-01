// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "optimizer_api.h"
#include "core/graph/graph.h"
#include "core/framework/execution_provider.h"
#include "core/framework/transform_layout_functions.h"

namespace onnxruntime {
/// <summary>
/// Creates concrete implementation of api for transpose optimizer. IMPORTANT: graph must have up-to-date edges,
///   node_arg-to-producer, and node_arg-to-consumer relationships. Otherwise call Resolve() before this.
/// </summary>
/// <param name="graph">ORT Graph to wrap with API</param>
/// <param name="cpu_allocator">Allocator used for reshaping/transposing tensors</param>
/// <param name="new_node_ep">New nodes are assigned to this EP, or left unassigned if nullptr</param>
/// <returns>api::GraphRef for use with transpose optimizer</returns>
std::unique_ptr<onnx_layout_transformation::api::GraphRef> MakeApiGraph(onnxruntime::Graph& graph,
                                                                        AllocatorPtr cpu_allocator,
                                                                        const char* new_node_ep);

/// <summary>
/// Creates NodeRef.
/// </summary>
/// <param name="graph">ORT Graph which owns the node</param>
/// <param name="node">ORT Node to wrap with API.</param>
/// <returns>api::NodeRef for use with transpose optimizer</returns>
std::unique_ptr<onnx_layout_transformation::api::NodeRef> MakeApiNode(onnxruntime::Graph& graph, onnxruntime::Node& node);

/// <summary>
/// Reveals underlying ORT graph from an api::GraphRef
/// </summary>
/// <param name="graph">api:GraphRef created from MakeApiGraph</param>
/// <returns>ORT graph</returns>
onnxruntime::Graph& GraphFromApiGraph(onnx_layout_transformation::api::GraphRef& graph);

/// <summary>
/// Reveals underlying ORT node from an api::NodeRef
/// </summary>
/// <param name="graph">api:NodeRef from graph created from MakeApiGraph</param>
/// <returns>ORT node</returns>
onnxruntime::Node& NodeFromApiNode(onnx_layout_transformation::api::NodeRef& node);

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
onnx_layout_transformation::CostCheckResult OrtEPCostCheck(
    const onnx_layout_transformation::api::GraphRef& graph,
    const onnx_layout_transformation::api::NodeRef& node,
    const std::vector<int64_t>& perm,
    const std::unordered_set<std::string>& outputs_leading_to_transpose);

namespace layout_transformer {
/// <summary>
/// Gets a list of layout sensitive ops for ORT. This list contains onnx standard defined
/// layout senstive ops + contrib ops + ops which are not layout sensitive but are treated as
/// layout sensitive by ORT EPs (exmaple Resize).
/// </summary>
/// <returns>unordered set of op_types which are layout sensitive</returns>
const std::unordered_set<std::string_view>& GetORTLayoutSensitiveOps();

/// <summary>
/// Transforms data layout from NCHW to NHWC, using the kMSInternalNHWCDomain domain for updated nodes.
///
/// This can be used by a compiling EP such as NNAPI, where the synthetic domain is a signal that the node has been
/// updated to the EP's required layout, or an EP with statically registered kernels such as XNNPACK where a kernel
/// is registered for the NHWC version of an ONNX operator. The NHWC version of the ONNX operator uses the synthetic
/// domain and is defined by onnxruntime/core/graph/contrib_ops/internal_nhwc_onnx_opset.cc
///
/// Transforms are applied to layout sensitive nodes assigned to execution_provider provided by the caller,
/// and any other non-layout sensitive nodes in order to optimize the transposes as much as possible.
/// </summary>
/// <param name="graph">graph to transform</param>
/// <param name="modified">indicates whether the graph is modified during transformation</param>
/// <param name="execution_provider">execution provider for which the transformation needs to be performed</param>
/// <param name="cpu_allocator">a CPU allocator used in layout transformation.
/// <param name="debug_graph_fn">Optional functor to debug the graph produced during layout transformation.
/// This is called after layout transformation if new nodes are inserted, and again after those are optimized.
/// </param>
Status TransformLayoutForEP(Graph& graph, bool& modified, const IExecutionProvider& execution_provider,
                            AllocatorPtr cpu_allocator, const DebugGraphFn& debug_graph_fn = {});

/// <summary>
/// Checks if the opset of the Graph is supported by the layout transformer.
/// </summary>
/// <param name="graph">Graph to check</param>
/// <returns></returns>
bool IsSupportedOpset(const Graph& graph);
}  // namespace layout_transformer
}  // namespace onnxruntime
