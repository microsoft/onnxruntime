// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/graph/basic_types.h>

namespace onnxruntime {

class GraphViewer;
class NodeArg;

namespace logging {
class Logger;
}

namespace coreml {

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by CoreML EP
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const logging::Logger& logger);

// CoreML is more efficient running using Apple Neural Engine
// This is to detect if the current system has Apple Neural Engine
bool HasNeuralEngine(const logging::Logger& logger);

}  // namespace coreml
}  // namespace onnxruntime
