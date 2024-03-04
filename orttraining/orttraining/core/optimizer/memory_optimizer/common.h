// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/graph/basic_types.h"
#include "core/framework/data_types.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime::optimizer::memory_optimizer {

// Uncomment for debugging Memory optimizer (MO).
// #define MO_NEED_LOG_DEBUG_INFO 1

#ifndef MO_LOG_DEBUG_INFO
#ifdef MO_NEED_LOG_DEBUG_INFO
#define MO_LOG_DEBUG_INFO(logger, message) LOGS(logger, WARNING) << message
#else
#define MO_LOG_DEBUG_INFO(logger, message) LOGS(logger, VERBOSE) << message
#endif
#endif

using NodeOutputPort = std::pair<const Node*, size_t>;
using ActivationUsedMap = InlinedHashMap<std::string, std::pair<bool, bool>>;

/**
 * @brief Type of memory reduction techniques.
 */
enum class OptimizationType {
  None = 0,  // Disabled.
  Recompute = 1,
  RecomputeWithCompromise = 2,
  TypeMax = 3,
};

std::string OptimizationTypeToString(OptimizationType type);

/**
 * @brief Type of user config.
 * type: type of memory reduction techniques.
 * requested_count: the number of occurrences of a subgraph pattern for alleviation. -1 means apply all.
 *   One example: if a subgraph pattern is found 3 times, and requested_count is set 2, then the 1st and 2nd subgraph
 *   in topological order will be applied for alleviation. This is useful to avoid alleviating more memory than
 *   needed.
 */
struct UserConfig {
  OptimizationType type;
  int requested_count;
};

/**
 * @brief Get total element count inn format of a symbolic string.
 * Be noted: this function is used to generate a unique string for a tensor shape.
 * For empty dim param, it is possible to have different symbolic string for the same shape, because there is
 * a static index_empty_dim used to generate empty dim param as a string.
 *
 * @param node The node to get element count.
 * @param output_index The output index of the node.
 * @return std::string
 */
std::string GetTensorElemCountInSymbolicString(const Node* node, size_t output_index);

int ParseIntValueFromString(std::string_view str);

Status ParseOptimizationConfigFromString(std::string_view memory_optimization_config,
                                         InlinedHashMap<std::string, UserConfig>& cluster_id_to_config_map);

}  // namespace onnxruntime::optimizer::memory_optimizer
