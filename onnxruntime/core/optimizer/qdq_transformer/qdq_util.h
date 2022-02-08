// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>

#include <gsl/gsl>

#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/optimizer/qdq_transformer/qdq_util_minimal.h"

namespace onnxruntime {

class Graph;
class Node;

namespace logging {
class Logger;
}

namespace QDQ {

// Check Q node op type, version, and domain.
bool MatchQNode(const Node& node);

// Check DQ node op type, version, and domain.
bool MatchDQNode(const Node& node);

/** Remove redundant DQ/Q pairs from the graph.
 * In particular, a DQ -> Q where the quantization parameters and types are the same has no effect.
 * Does not handle subgraphs.
 * @param graph The graph to update.
 * @param node_indices The graph node indices. This should cover the whole graph.
 *                     It is provided as an argument to allow reuse of an existing node index list.
 * @param compatible_providers Any compatible execution providers. Empty means all are compatible.
 * @param logger A logger instance.
 * @param[out] modified Set to true if the graph was modified.
 */
common::Status CancelOutRedundantDQQPairs(Graph& graph, gsl::span<const NodeIndex> node_indices,
                                          const std::unordered_set<std::string>& compatible_providers,
                                          const logging::Logger& logger,
                                          bool& modified);

}  // namespace QDQ
}  // namespace onnxruntime
