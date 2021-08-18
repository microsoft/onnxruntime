// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/graph/basic_types.h"

namespace onnxruntime::experimental::utils {

/**
 * Gets the key that can be used to look up a fbs::SubGraphSessionState in a fbs::SessionState.
 *
 * @param node_idx The index of the node in the current graph.
 * @param attr_name The name of the node attribute that contains the subgraph.
 * @return The subgraph key.
 */
inline std::string GetSubGraphId(const NodeIndex node_idx, const std::string& attr_name) {
  return std::to_string(node_idx) + "_" + attr_name;
}

}  // namespace onnxruntime::experimental::utils
