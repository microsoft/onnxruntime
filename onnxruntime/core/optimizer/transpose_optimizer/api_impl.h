// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/graph.h"
#include "core/optimizer/transpose_optimizer/api.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

/// <summary>
/// Creates concrete implementation of api for transpose optimizer
/// </summary>
/// <param name="graph">ORT Graph to wrap with API</param>
/// <param name="cpu_allocator">Allocator used for reshaping/transposing tensors</param>
/// <param name="logger">Logger</param>
/// <param name="new_node_ep">New nodes are assigned to this EP, or left unassigned if nullptr</param>
/// <returns>api::Graph for use with transpose optimizer</returns>
std::unique_ptr<onnx_layout_transformation::api::Graph> MakeApiGraph(onnxruntime::Graph& graph,
                                                                     AllocatorPtr cpu_allocator,
                                                                     const logging::Logger& logger,
                                                                     const char* new_node_ep);

/// <summary>
/// Reveals underlying ORT graph from an api::Graph
/// </summary>
/// <param name="graph">api:Graph created from MakeApiGraph</param>
/// <returns>ORT graph</returns>
onnxruntime::Graph& GraphFromApiGraph(onnx_layout_transformation::api::Graph& graph);

/// <summary>
/// Reveals underlying ORT node from an api::Node
/// </summary>
/// <param name="graph">api:Node from graph created from MakeApiGraph</param>
/// <returns>ORT node</returns>
onnxruntime::Node& NodeFromApiNode(onnx_layout_transformation::api::Node& node);

}  // namespace onnxruntime
