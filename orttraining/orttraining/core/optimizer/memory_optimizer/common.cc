// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <charconv>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/tensorprotoutils.h"

#include "core/common/string_utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

namespace {

/**
 * @brief Prepare info including activation usage, node usage in fw and bw.
 *
 * @param graph Graph to iterate.
 * @param boundary_op_order_in_topological_sort index of the boundary op between fw and bw.
 * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
 * @param fw_op_output_arg_used_map Collected activation usage mapping.
 *   - key: node arg name
 *   - value: a pair of bool, representing whether the activation is used by forward nodes or by backward nodes.
 * @param is_forward_nodes Collected node is forward pass op mapping.
 */
void GetForwardOutputUsageMap(const Graph& graph,
                              const ptrdiff_t boundary_op_order_in_topological_sort,
                              const InlinedHashMap<NodeIndex, size_t>&
                                  node_index_to_its_order_in_topological_sort_map,
                              InlinedHashMap<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map,
                              InlinedHashMap<const Node*, bool>& is_forward_nodes) {
  fw_op_output_arg_used_map.clear();
  ORT_ENFORCE(boundary_op_order_in_topological_sort >= 0);
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();
  is_forward_nodes.reserve(node_ids.size());

  auto is_forward_pass_operator = [](ptrdiff_t op_order_in_topological_sort,
                                     ptrdiff_t boundary_op_order_in_topological_sort) -> bool {
    return op_order_in_topological_sort <= boundary_op_order_in_topological_sort;
  };

  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr /* skip removed nodes*/) {
      continue;
    }

    const Node& node = *p_node;

    bool is_forward_op = is_forward_pass_operator(static_cast<ptrdiff_t>(i), boundary_op_order_in_topological_sort);
    if (!is_forward_op) {
      is_forward_nodes[p_node] = false;
      continue;
    }

    is_forward_nodes[p_node] = true;

    for (auto& output_arg : node.OutputDefs()) {
      bool used_in_fw = false;
      bool used_in_bw = false;
      for (auto& consumer_node : graph.GetConsumerNodes(output_arg->Name())) {
        size_t consumer_node_index_in_topological_order =
            node_index_to_its_order_in_topological_sort_map.at(consumer_node->Index());
        if (is_forward_pass_operator(static_cast<ptrdiff_t>(consumer_node_index_in_topological_order),
                                     boundary_op_order_in_topological_sort)) {
          used_in_fw = true;
        } else {
          used_in_bw = true;
        }
      }
      fw_op_output_arg_used_map.insert({{output_arg->Name(), std::make_pair(used_in_fw, used_in_bw)}});
    }
  }
}

// trim from start (in place)
static inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// trim from end (in place)
static inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
}

// trim from both ends
static inline std::string trim_copy(std::string s) {
  rtrim(s);
  ltrim(s);
  return s;
}

std::string empty_dim_param_placeholder = "empty_dim_param";
static int64_t index_empty_dim = 0;

void TensorShapeProtoToDimParamVector(const ONNX_NAMESPACE::TensorShapeProto* shape,
                                      std::vector<std::string>& dim_params,
                                      bool& has_unknown_dim) {
  has_unknown_dim = false;
  for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
    auto dim = shape->dim(dim_index);
    if (utils::HasDimValue(dim)) {
      dim_params.push_back(std::to_string(dim.dim_value()));

    } else {
      std::string trimmed_dim_param = trim_copy(dim.dim_param());
      if (trimmed_dim_param.empty()) {
        has_unknown_dim = true;
        dim_params.push_back(empty_dim_param_placeholder + std::to_string(index_empty_dim++));
      } else {
        dim_params.push_back(trimmed_dim_param);
      }
    }
  }

  if (shape->dim_size() == 0) {
    dim_params.push_back("(1)");  // Scalar
  }
}

bool HasUnknowDimension(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (shape == nullptr) {
    return true;
  }

  std::vector<std::string> dim_params;
  bool has_unknown_dim = false;
  TensorShapeProtoToDimParamVector(shape, dim_params, has_unknown_dim);

  return has_unknown_dim;
}

std::string TensorShapeProtoToString(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (shape == nullptr) {
    return "unknown";
  }

  std::vector<std::string> dim_params;
  bool has_unknown_dim = false;
  TensorShapeProtoToDimParamVector(shape, dim_params, has_unknown_dim);

  std::ostringstream oss;
  oss << "(";
  for (auto it = dim_params.begin(); it != dim_params.end(); ++it) {
    oss << "(" << *it << ")";
    if (it != (dim_params.end() - 1)) {
      oss << "*";
    }
  }
  oss << ")";

  return oss.str();
}

}  // namespace

std::string GetTensorElemCountInSymbolicString(const Node* node, int output_index) {
  const auto& output_def = node->OutputDefs()[output_index];
  const auto shape = output_def->Shape();

  std::string shape_str = TensorShapeProtoToString(shape);

  // If the output shape contains unknown dimension, we try to get the shape from input.
  // though the input shape might be different, but its elem size and count should be the same
  // with the output.
  if (node->OpType() == "Reshape" && HasUnknowDimension(shape) &&
      !HasUnknowDimension(node->InputDefs()[0]->Shape())) {
    shape_str = TensorShapeProtoToString(node->InputDefs()[0]->Shape());
  }

  return shape_str;
}

Status GetStashedActivationCandidates(const Graph& graph,
                                      const ptrdiff_t boundary_op_order_in_topological_sort,
                                      InlinedHashMap<std::string, std::pair<bool, bool>>&
                                          fw_op_output_arg_used_map,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                          candidate_output_args_map,
                                      InlinedHashMap<const Node*, bool>& is_forward_nodes,
                                      const logging::Logger& logger) {
  if (boundary_op_order_in_topological_sort < 0) {
    LOGS(logger, VERBOSE) << "No boundary op found. Skip memory optimization.";
    return Status::OK();
  }

  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  InlinedHashMap<NodeIndex, size_t> node_index_to_its_order_in_topological_sort_map;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    node_index_to_its_order_in_topological_sort_map[p_node->Index()] = i;
  }

  GetForwardOutputUsageMap(graph, boundary_op_order_in_topological_sort,
                           node_index_to_its_order_in_topological_sort_map,
                           fw_op_output_arg_used_map,
                           is_forward_nodes);

  for (auto& kv : fw_op_output_arg_used_map) {
    // used by fw and bw, then it is a candidate.
    if (kv.second.first && kv.second.second) {
      const Node* n = graph.GetProducerNode(kv.first);
      ORT_ENFORCE(n, "Activation should have a producer node");
      size_t k = 0;
      for (k = 0; k < n->OutputDefs().size(); ++k) {
        if (n->OutputDefs()[k]->Name().compare(kv.first) == 0) {
          break;
        }
      }

      if (std::find(candidate_output_args_map[n].begin(), candidate_output_args_map[n].end(), k) !=
          candidate_output_args_map[n].end()) {
        ORT_ENFORCE(false, "Duplicated candidate output found.");
      }

      candidate_output_args_map[n].push_back(k);
      LOGS(logger, VERBOSE) << "Find candidate output named [" << kv.first << "] of Node " << n->Name() << "("
                            << n->OpType() << ")";
    }
  }

  return Status::OK();
}

void NodesInTopoOrderToString(const InlinedVector<const Node*>& nodes_in_topological_order,
                              std::string& subgraph_string_representation,
                              std::string& log_info) {
  std::ostringstream oss;
  std::ostringstream subgraph_string_representation_oss;
  size_t node_count = nodes_in_topological_order.size();
  for (size_t i = 0; i < node_count; ++i) {
    if (i < node_count - 1) {  // Ignore the last node.
      oss << "(name:" << nodes_in_topological_order[i]->Name() << ", type:" << nodes_in_topological_order[i]->OpType()
          << "),";
    }

    subgraph_string_representation_oss << nodes_in_topological_order[i]->OpType() << "+";
  }

  subgraph_string_representation = subgraph_string_representation_oss.str();
  log_info = oss.str();
  if (log_info.size() > 0) {
    log_info = " with its precedent nodes: " + log_info;
  }
}

std::string OptimizationTypeToString(OptimizationType type) {
  switch (type) {
    case OptimizationType::None:
      return "None";
    case OptimizationType::Recompute:
      return "Recompute";
    case OptimizationType::RecomputeWithCompromise:
      return "RecomputeWithCompromise";
    default:
      ORT_THROW("Unknown optimization type.");
  }
}

int ParseIntValueFromString(std::string_view str) {
  int int_value = 0;
  auto result = std::from_chars(str.data(), str.data() + str.size(), int_value);
  ORT_ENFORCE(result.ec != std::errc::invalid_argument, "Fail to convert to int from string: ", str);
  return int_value;
}

Status ParseConfigFromString(const std::string memory_optimization_config,
                             InlinedHashMap<std::string, UserConfig>& cluster_id_to_config_map) {
  if (!memory_optimization_config.empty()) {
    const auto user_config_strs = utils::SplitString(memory_optimization_config, ",");
    for (const auto& user_config_str : user_config_strs) {
      const auto user_config = utils::SplitString(user_config_str, ":");
      ORT_RETURN_IF_NOT(user_config.size() == 3,
                        "User config should be in format of SubgraphStr:OptimizationType:RequestApplyCount.");

      const std::string subgraph_string_representation(user_config[0]);
      int optimization_type_int = ParseIntValueFromString(user_config[1]);
      int requested_apply_count = ParseIntValueFromString(user_config[2]);
      ORT_RETURN_IF_NOT(optimization_type_int <
                                static_cast<int>(OptimizationType::TypeMax) &&
                            optimization_type_int >= 0,
                        "Invalid optimization type specified for subgraph: ",
                        subgraph_string_representation);

      ORT_RETURN_IF_NOT(requested_apply_count == -1 || requested_apply_count >= 0,
                        "Invalid requested_apply_count specified for subgraph: ", requested_apply_count);

      // At this point, subgraph_string_representation is a pattern graph string representation.
      // If duplicated subgraph_string_representation is found in user config, the last one will be used.
      cluster_id_to_config_map[subgraph_string_representation] = UserConfig{
          static_cast<OptimizationType>(optimization_type_int),
          requested_apply_count};
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime::optimizer::memory_optimizer
