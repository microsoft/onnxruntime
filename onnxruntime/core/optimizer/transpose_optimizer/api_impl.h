// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/graph.h"
#include "core/optimizer/transpose_optimizer/api.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

std::unique_ptr<onnx_layout_transformation::api::Graph> MakeApiGraph(onnxruntime::Graph& graph,
                                                                     AllocatorPtr cpu_allocator,
                                                                     const logging::Logger& logger,
                                                                     const char* new_node_ep);

// Reveal API internals
onnxruntime::Graph& GraphFromApiGraph(onnx_layout_transformation::api::Graph& graph);

onnxruntime::Node& NodeFromApiNode(onnx_layout_transformation::api::Node& node);

}  // namespace onnxruntime
