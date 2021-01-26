// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {

class GraphViewer;
class NodeArg;

namespace logging {
class Logger;
}

namespace coreml {

// TODO, move this to shared_library
template <template <typename> class Container, typename T>
T Product(const Container<T>& c) {
  return static_cast<T>(accumulate(c.cbegin(), c.cend(), 1, std::multiplies<T>()));
}

// TODO, move this to shared_library
template <class Map, class Key>
inline bool Contains(const Map& map, const Key& key) {
  return map.find(key) != map.end();
}

// TODO, move this to shared_library
bool GetType(const NodeArg& node_arg, int32_t& type, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by CoreML EP
std::vector<std::vector<size_t>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                   const logging::Logger& logger);

}  // namespace coreml
}  // namespace onnxruntime
