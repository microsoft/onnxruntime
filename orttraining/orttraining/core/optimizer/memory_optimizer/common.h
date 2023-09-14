// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

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
#define MO_LOG_DEBUG_INFO(logger, message) \
  ORT_UNUSED_PARAMETER(logger);            \
  do {                                     \
  } while (0)
#endif
#endif

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
 * @param graph_viewer Graph to iterate.
 * @param boundary_op_order_in_topological_sort The order of the boundary op in the topological sort.
 * @param fw_op_output_arg_used_map Activation usage mapping.
 * @param candidate_output_args_map Candidate activations, which are consumed by both fw and bw ops.
 * @param is_forward_nodes Whether a node is a forward node.
 * @param logger Logger.
 * @return Status
 */
Status GetStashedActivationCandidates(const GraphViewer& graph_viewer,
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
void NodesInTopoOrderToString(gsl::span<const Node* const> nodes_in_topological_order,
                              std::string& subgraph_string_representation,
                              std::string& log_info);

std::string GetTensorElemCountInSymbolicString(const Node* node, int output_index);

/**
 * @brief Struct to store properties of a specific subgraph.
 */
class ClusterApplyContext {
 public:
  ClusterApplyContext() = default;

  OptimizationType type;
  int requested_count{0};
  int total_frequency{0};  // The occurrence of this subgraph pattern in the graph.

  int applied_count{0};  // The number of times this subgraph pattern has been really applied in this transformer.
  int skip_count{0};     // The number of times this subgraph instance has been skipped in reversed topological order.
};

/**
 * @brief Base class for a concrete optimization plan.
 *
 */
class NodeOptimizationPlanBase {
 public:
  NodeOptimizationPlanBase(const Node* node,
                           gsl::span<const size_t> activation_output_indices,
                           float save_ratio)
      : node(node),
        activation_output_indices_(activation_output_indices.begin(), activation_output_indices.end()),
        save_ratio_(save_ratio) {
  }

  virtual ~NodeOptimizationPlanBase() = default;

  virtual OptimizationType GetOptimizationType() const = 0;
  virtual std::string GetClusterId() const = 0;
  virtual std::string NormalizeForNodeClusterId() const = 0;

  gsl::span<const size_t> GetActivationOutputIndices() const { return activation_output_indices_; }

  float GetSaveRatio() const { return save_ratio_; }

  std::string GetMemorySavingSymbolicString() const {
    std::string saving_str;
    for (auto output_index : activation_output_indices_) {
      // If the output is reusing other node's buffer, then no memory saving.
      if (reuse_buffers.find(output_index) != reuse_buffers.end()) {
        continue;
      }

      const auto& output_def = node->OutputDefs()[output_index];
      MLDataType ml_data_type = DataTypeImpl::TypeFromProto(*output_def->TypeAsProto());
      ORT_ENFORCE(ml_data_type->IsTensorType(), "ml_type must be a tensor type, but it is ",
                  DataTypeImpl::ToString(ml_data_type));
      const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
      ORT_ENFORCE(nullptr != tensor_type_base);
      MLDataType elt_type = tensor_type_base->GetElementType();
      const auto byte_count_per_element = elt_type->Size();
      if (!saving_str.empty()) {
        saving_str += " + ";
      }
      saving_str = "(" + GetTensorElemCountInSymbolicString(node, output_index) + " * " +
                   std::to_string(byte_count_per_element) + " * " +
                   std::to_string(GetSaveRatio()) + ")";
    }
    if (saving_str.empty()) {
      return saving_str;
    }
    return "(" + saving_str + ")";
  }

  const Node* node;
  InlinedVector<size_t> activation_output_indices_;
  float save_ratio_ = 1.0f;
  // A map: output index reusing other node's output (other_node, output index)
  InlinedHashMap<size_t, std::pair<const Node*, size_t>> reuse_buffers;
};

int ParseIntValueFromString(std::string_view str);

Status ParseConfigFromString(std::string_view memory_optimization_config,
                             InlinedHashMap<std::string, UserConfig>& cluster_id_to_config_map);

}  // namespace onnxruntime::optimizer::memory_optimizer
