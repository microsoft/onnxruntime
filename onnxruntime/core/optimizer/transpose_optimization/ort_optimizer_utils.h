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
std::unique_ptr<onnx_transpose_optimization::api::GraphRef> MakeApiGraph(onnxruntime::Graph& graph,
                                                                         AllocatorPtr cpu_allocator,
                                                                         const char* new_node_ep);

/// <summary>
/// Creates NodeRef.
/// </summary>
/// <param name="graph">ORT Graph which owns the node</param>
/// <param name="node">ORT Node to wrap with API.</param>
/// <returns>api::NodeRef for use with transpose optimizer</returns>
std::unique_ptr<onnx_transpose_optimization::api::NodeRef> MakeApiNode(onnxruntime::Graph& graph,
                                                                       onnxruntime::Node& node);

/// <summary>
/// Reveals underlying ORT graph from an api::GraphRef
/// </summary>
/// <param name="graph">api:GraphRef created from MakeApiGraph</param>
/// <returns>ORT graph</returns>
onnxruntime::Graph& GraphFromApiGraph(onnx_transpose_optimization::api::GraphRef& graph);

/// <summary>
/// Reveals underlying ORT node from an api::NodeRef
/// </summary>
/// <param name="graph">api:NodeRef from graph created from MakeApiGraph</param>
/// <returns>ORT node</returns>
onnxruntime::Node& NodeFromApiNode(onnx_transpose_optimization::api::NodeRef& node);
}  // namespace onnxruntime
