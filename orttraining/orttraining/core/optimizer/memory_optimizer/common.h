// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/inlined_containers.h"
#include "core/common/const_pointer_container.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/constants.h"
#include "core/graph/graph_nodes.h"
#include "core/graph/graph.h"

namespace onnxruntime::optimizer::memory_optimizer {

using NodeOutputPort = std::pair<const Node*, int>;
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
 * @brief Find all stashed activations, e.g. activations used by forward operators and backward operators.
 *
 * @param graph Graph to iterate.
 * @param boundary_op_order_in_topological_sort The order of the boundary op in the topological sort.
 * @param fw_op_output_arg_used_map Activation usage mapping.
 * @param candidate_output_args_map Candidate activations, which are consumed by both fw and bw ops.
 * @param is_forward_nodes Whether a node is a forward node.
 * @param logger Logger.
 * @return Status
 */
Status GetStashedActivationCandidates(const Graph& graph,
                                      const ptrdiff_t boundary_op_order_in_topological_sort,
                                      InlinedHashMap<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map,
                                      InlinedHashMap<const Node*, bool>& is_forward_nodes,
                                      const logging::Logger& logger);

/**
 * @brief Convert the recompute subgraph to its string representation.
 *
 * @param nodes_in_topological_order The subgraph nodes in topological order.
 * @param subgraph_string_representation Returns subgraph string representation.
 * @param log_info Returns log info for users.
 */
void NodesInTopoOrderToString(const InlinedVector<const Node*>& nodes_in_topological_order,
                              std::string& subgraph_string_representation,
                              std::string& log_info);

std::string GetTensorElemCountInSymbolicString(const Node* node, int output_index);

/**
 * @brief Struct to store properties of a specific subgraph.
 */
class ClusterApplyContext {
 public:
  ClusterApplyContext() = default;

  // virtual ~ClusterApplyContext() = default;

  OptimizationType type;
  int requested_count{0};
  int total_frequency{0};  // The occurrence of this subgraph pattern in the graph.

  int applied_count{0};  // The number of times this subgraph pattern has been really applied in this transformer.
  int skip_count{0};     // The number of times this subgraph instance has been skipped in reversed topological order.
};

class NodeOptimizationPlanBase {
 public:
  NodeOptimizationPlanBase(const Node* node) : node(node){

                                               };
  virtual ~NodeOptimizationPlanBase() = default;

  virtual OptimizationType GetOptimizationType() const = 0;
  virtual std::string GetClusterId() const = 0;
  virtual std::string NormalizeForNodeClusterId() const = 0;
  virtual const InlinedVector<size_t>& GetActivationOutputIndices() const = 0;

  // A map: output index reusing other node's output (other_node, output index)
  InlinedHashMap<size_t, std::pair<const Node*, size_t>> reuse_buffers;
  const Node* node;
};

int ParseIntValueFromString(std::string_view str);

Status ParseConfigFromString(const std::string memory_optimization_config,
                             InlinedHashMap<std::string, UserConfig>& cluster_id_to_config_map);

}  // namespace onnxruntime::optimizer::memory_optimizer
