// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
#include <vector>
#include <utility>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/tensorprotoutils.h"

#include "core/common/string_utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

namespace {

constexpr const char empty_dim_param_placeholder[] = "empty_dim_param";
static size_t index_empty_dim = 0;

bool TensorShapeProtoToDimParamVector(const ONNX_NAMESPACE::TensorShapeProto* shape,
                                      std::vector<std::string>& dim_params) {
  bool has_unknown_dim = false;
  for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
    auto dim = shape->dim(dim_index);
    if (utils::HasDimValue(dim)) {
      dim_params.push_back(std::to_string(dim.dim_value()));
    } else {
      std::string trimmed_dim_param = utils::TrimString(dim.dim_param());
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

  return has_unknown_dim;
}

bool HasUnknowDimension(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (shape == nullptr) {
    return true;
  }

  std::vector<std::string> dim_params;
  return TensorShapeProtoToDimParamVector(shape, dim_params);
}

std::string TensorShapeProtoToString(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (shape == nullptr) {
    return "unknown";
  }

  std::vector<std::string> dim_params;
  TensorShapeProtoToDimParamVector(shape, dim_params);

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

std::string GetTensorElemCountInSymbolicString(const Node* node, size_t output_index) {
  const auto& output_def = node->OutputDefs()[output_index];
  const auto shape = output_def->Shape();

  std::string shape_str = TensorShapeProtoToString(shape);

  // If the output shape contains an unknown dimension, we try to get the shape from the input.
  // Though the input shape might be different, its elem size and count should be the same
  // with the output.
  if (node->OpType() == "Reshape" && HasUnknowDimension(shape) &&
      !HasUnknowDimension(node->InputDefs()[0]->Shape())) {
    shape_str = TensorShapeProtoToString(node->InputDefs()[0]->Shape());
  }

  return shape_str;
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

Status ParseOptimizationConfigFromString(std::string_view memory_optimization_config,
                                         InlinedHashMap<std::string, UserConfig>& cluster_id_to_config_map) {
  if (!memory_optimization_config.empty()) {
    const auto user_config_strs = utils::SplitString(memory_optimization_config, ",");
    for (const auto& user_config_str : user_config_strs) {
      const auto user_config = utils::SplitString(user_config_str, ":");
      ORT_RETURN_IF_NOT(user_config.size() == 3,
                        "User config should be in the format of SubgraphStr:OptimizationType:RequestApplyCount.");

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
      // If a duplicated subgraph_string_representation is found in user config, the last one will be used.
      cluster_id_to_config_map[subgraph_string_representation] = UserConfig{
          static_cast<OptimizationType>(optimization_type_int),
          requested_apply_count};
    }
  }

  return Status::OK();
}

void SortNodesInTopoOrder(const InlinedHashMap<NodeIndex, ptrdiff_t>&
                              node_index_to_its_order_in_topological_sort_map,
                          InlinedVector<const Node*>& nodes) {
  std::sort(nodes.begin(), nodes.end(),
            [&node_index_to_its_order_in_topological_sort_map](const Node*& lhs, const Node*& rhs) {
              return node_index_to_its_order_in_topological_sort_map.at(lhs->Index()) <
                     node_index_to_its_order_in_topological_sort_map.at(rhs->Index());
            });
}

Status FindReachableNodesFromGivenInputAndOutput(const GraphViewer& graph_viewer,
                                                 const Node* start_node_inclusive,
                                                 const Node* end_node_exclusive,
                                                 InlinedVector<const Node*>& reachable_nodes,
                                                 const Node*& end_node_input_node,
                                                 int32_t& output_index) {
  // Reverse BFS from end_node_exclusive by its input edges to find all reachable inputs nodes, skip if it hit the
  // start_node_inclusive.
  std::deque<const Node*> nodes_to_check;
  std::set<const Node*> visited_nodes;

  const NodeArg* input_arg = end_node_exclusive->InputDefs()[0];
  ORT_ENFORCE(input_arg->Exists(), "Input arg should exist.");
  const Node* input_node = graph_viewer.GetProducerNode(input_arg->Name());
  end_node_input_node = input_node;
  output_index = optimizer_utils::IndexOfNodeOutput(*end_node_input_node, *input_arg);

  nodes_to_check.push_back(end_node_input_node);
  InlinedHashSet<const Node*> reachable_input_nodes;

  while (!nodes_to_check.empty()) {
    const Node* next_node = nodes_to_check.front();
    nodes_to_check.pop_front();

    if (visited_nodes.find(next_node) != visited_nodes.end()) {
      continue;
    }

    visited_nodes.insert(next_node);
    if (next_node == start_node_inclusive) {
      continue;
    }

    for (size_t input_index = 0; input_index < next_node->InputDefs().size(); ++input_index) {
      const NodeArg* input_arg = next_node->InputDefs()[input_index];
      if (!input_arg->Exists()) {
        continue;
      }
      const Node* input_node = graph_viewer.GetProducerNode(input_arg->Name());
      if (input_node != nullptr) {
        nodes_to_check.push_back(input_node);
        reachable_input_nodes.insert(input_node);
      }
    }
  }

  std::cout << "Foud reachable input nodes: " << reachable_input_nodes.size() << "\n";

  // BFS from start_node_inclusive by its output edges to find all reachable output nodes, skip if it is not in the
  // reachable_input_nodes.
  nodes_to_check.clear();
  visited_nodes.clear();
  nodes_to_check.push_back(start_node_inclusive);

  while (!nodes_to_check.empty()) {
    const Node* next_node = nodes_to_check.front();
    nodes_to_check.pop_front();

    if (visited_nodes.find(next_node) != visited_nodes.end()) {
      continue;
    }

    visited_nodes.insert(next_node);
    if (reachable_input_nodes.find(next_node) == reachable_input_nodes.end()) {
      continue;
    }

    reachable_nodes.push_back(next_node);
    for (size_t output_index = 0; output_index < next_node->OutputDefs().size(); ++output_index) {
      const NodeArg* output_arg = next_node->OutputDefs()[output_index];
      if (!output_arg->Exists()) {
        continue;
      }
      for (auto& consumer_node : graph_viewer.GetConsumerNodes(output_arg->Name())) {
        nodes_to_check.push_back(consumer_node);
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime::optimizer::memory_optimizer
