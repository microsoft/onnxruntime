// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"

#include "core/providers/coreml/shape_utils.h"  // TODO add this include as needed and remove from this file

namespace onnxruntime {

class GraphViewer;
class NodeArg;
class Node;

namespace logging {
class Logger;
}

namespace coreml {

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const logging::Logger& logger);

// Gets the set of nodes that are supported by the CoreML EP.
std::unordered_set<const Node*> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                  const logging::Logger& logger);

// CoreML is more efficient running using Apple Neural Engine
// This is to detect if the current system has Apple Neural Engine
bool HasNeuralEngine(const logging::Logger& logger);

}  // namespace coreml
}  // namespace onnxruntime
