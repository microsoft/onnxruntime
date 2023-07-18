// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {

class GraphViewer;
class NodeArg;
class Node;

namespace logging {
class Logger;
}

namespace coreml {

// Gets `node_arg`'s shape. Dynamic dimensions will have a value of -1.
bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

// Gets `node_arg`'s shape if it has no dynamic dimensions.
bool GetStaticShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

// True iff `shape` has no dynamic dimensions.
bool IsStaticShape(gsl::span<const int64_t> shape);

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
