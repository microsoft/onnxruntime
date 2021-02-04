// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>

namespace onnxruntime {

class GraphViewer;
class NodeArg;

namespace logging {
class Logger;
}

namespace coreml {

common::Status GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape);

// TODO, move this to shared_library
bool GetType(const NodeArg& node_arg, int32_t& type, const logging::Logger& logger);

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by CoreML EP
std::vector<std::vector<size_t>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                   const logging::Logger& logger);

}  // namespace coreml
}  // namespace onnxruntime
