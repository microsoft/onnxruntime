// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/graph/ep_api_types.h"



namespace onnxruntime {
namespace ep_graph_utils {
/// <summary>
/// Adds a new initializer to 'graph' with new_initializer that points to the OrtValue buffer
/// </summary>
/// <param name="graph">target graph</param>
/// <param name="new_initializer">TensorProto with external data contained in ort_value</param>
/// <param name="ort_value">ort_value with data</param>
/// <returns></returns>
OrtStatusPtr GetSubGraphAsModelFromGraph(const OrtGraph* src_graph,
                                         const OrtNode** nodes,
                                         size_t num_nodes,
                                         bool copy_in_memory_initializer,
                                         std::unique_ptr<Model>& out_model);
}  // namespace ep_graph_utils

}  // namespace onnxruntime
