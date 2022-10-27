// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/optimizer/memory_alleviation.h"

namespace onnxruntime {

namespace {

constexpr int32_t MAXIMUM_RECOMPUTE_NODE_COUNT = 15;

std::string TensorShapeProtoToString(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  std::ostringstream shape_oss;
  if (shape != nullptr) {
    for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
      auto dim = shape->dim(dim_index);
      if (utils::HasDimValue(dim)) {
        shape_oss << dim.dim_value() << " x ";
      } else {
        shape_oss << dim.dim_param() << " x ";
      }
    }
  } else {
    shape_oss << "unknown";
  }

  return shape_oss.str();
}

int ParseIntValueFromString(std::string_view str) {
  int int_value = 0;
  auto result = std::from_chars(str.data(), str.data() + str.size(), int_value);
  ORT_ENFORCE(result.ec != std::errc::invalid_argument, "Fail to convert to int from string: ", str);
  return int_value;
}

bool IsForwardPassOperator(int64_t op_order_in_topological_sort, int64_t boundary_op_order_in_topological_sort) {
  return op_order_in_topological_sort <= boundary_op_order_in_topological_sort;
}

static size_t GetElementSize(const ONNX_NAMESPACE::DataType& tensor_type) {
  const ONNX_NAMESPACE::TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(tensor_type);
  MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
  const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
  ORT_ENFORCE(nullptr != tensor_type_base);
  MLDataType elt_type = tensor_type_base->GetElementType();
  return elt_type->Size();
}

// TODO(pengwa): extend this function to be more general.
float InputOutputSizeRatio(const Node* node) {
  if (node->OpType().compare("Cast") == 0) {
    const NodeArg* input = node->InputDefs()[0];
    const NodeArg* output = node->OutputDefs()[0];
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING ||
        output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      return 1.0f;
    }
    const auto& ptype1 = input->Type();
    const auto& ptype2 = output->Type();
    float ratio = float(GetElementSize(ptype1)) / (float)GetElementSize(ptype2);
    return ratio;
  }

  return 1.0f;
}

}  // namespace

Status MemoryAlleviation::ParseConfigFromString(const std::string& enable_memory_alleviation,
                                                const std::string& level) {
  memory_alleviation_config_ = enable_memory_alleviation;
  if (!enable_memory_alleviation.empty()) {
    const auto alleviation_map = utils::SplitString(enable_memory_alleviation, ",");
    for (const auto& alleviation_config_per_op_str : alleviation_map) {
      const auto alleviation_config_per_op = utils::SplitString(alleviation_config_per_op_str, ":");
      ORT_RETURN_IF_NOT(alleviation_config_per_op.size() == 3,
                        "Alleviation config for each operator should in format of OpType:AlleviationType.");

      const std::string subgraph_string_representation(alleviation_config_per_op[0]);
      int alleviation_type_int = ParseIntValueFromString(alleviation_config_per_op[1]);
      int alleviation_freq = ParseIntValueFromString(alleviation_config_per_op[2]);
      ORT_RETURN_IF_NOT(alleviation_type_int < 2 && alleviation_type_int >= 0,
                        "Invalid alleviation type specified for subgraph: ", subgraph_string_representation);

      ORT_RETURN_IF_NOT(alleviation_freq == -1 || alleviation_freq >= 0,
                        "Invalid alleviation_freq specified for subgraph: ", alleviation_freq);

      // At this point, subgraph_string_representation is a pattern graph string representation.
      pattern_subgraph_to_user_alleviation_config_map_[subgraph_string_representation] =
          UserAlleviationConfig{static_cast<AlleviationType>(alleviation_type_int), alleviation_freq};
    }
  }

  int probe_level = ParseIntValueFromString(level);
  ORT_RETURN_IF_NOT(probe_level < 2 && probe_level >= 0,
                    "Invalid probe level specified: ", level);
  level_ = static_cast<ProbeLevel>(probe_level);

  return Status::OK();
}

MemoryAlleviation::MemoryAlleviation(const std::string& enable_memory_alleviation, const std::string& level)
    : GraphTransformer("MemoryAlleviation") {
  // Parse user defined alleviation configs.
  ORT_ENFORCE(ParseConfigFromString(enable_memory_alleviation, level).IsOK());

  if (static_cast<int>(level_) >= static_cast<int>(ProbeLevel::Basic)) {
    recomputable_op_type_to_input_arg_index_map_.insert({
        // Binary elementwise
        {"Add", EntryOperatorConfig{{0, 1}}},
        {"BiasGelu", EntryOperatorConfig{{0, 1}}},
        {"Div", EntryOperatorConfig{{0, 1}}},
        {"Mul", EntryOperatorConfig{{0, 1}}},
        {"Sub", EntryOperatorConfig{{0, 1}}},

        // Data layout
        /// The shape input is trivial whether it exists or not in backward.
        {"Reshape", EntryOperatorConfig{{0}}},
        {"Squeeze", EntryOperatorConfig{{0}}},
        {"Unsqueeze", EntryOperatorConfig{{0}}},

        // Unary elementwise
        /// The ratio and mode input are trivial whether they exists or not in backward
        {"BitmaskDropout", EntryOperatorConfig{{0}}},
        /// The axis input are trivial whether they exists or not in backward
        {"CumSum", EntryOperatorConfig{{0}}},
        {"Dropout", EntryOperatorConfig{{0}}},
        {"Gelu", EntryOperatorConfig{{0}}},
        {"FastGelu", EntryOperatorConfig{{0}}},

        // Tenary elementwise
        {"Where", EntryOperatorConfig{{0, 1, 2}}},

        // Data copy
        {"Tile", EntryOperatorConfig{{0}}},
        {"Cast", EntryOperatorConfig{{0}}},
    });
  }

  if (static_cast<int>(level_) >= static_cast<int>(ProbeLevel::Advanced)) {
    recomputable_op_type_to_input_arg_index_map_.insert({
        {"MatMul", EntryOperatorConfig{{0, 1}}},
        {"FusedMatMul", EntryOperatorConfig{{0, 1}}},
        {"Softmax", EntryOperatorConfig{{0}}},
        {"BiasSoftmax", EntryOperatorConfig{{0, 1}}},
        {"BiasSoftmaxDropout", EntryOperatorConfig{{0, 1}}},
    });
  }
}

int64_t MemoryAlleviation::PrepareForTransformation(const Graph& graph,
                                                    ActivationUsedMap& fw_op_output_arg_used_map,
                                                    InlinedHashMap<NodeIndex, size_t>&
                                                        node_index_to_its_order_in_topological_sort_map) const {
  fw_op_output_arg_used_map.clear();

  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Find boundary ops between forward and backward pass, currently, it's limited to YieldOp.
  int64_t yield_op_order_in_topological_sort = -1;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    if (p_node->OpType() == "YieldOp") {
      yield_op_order_in_topological_sort = static_cast<int64_t>(i);
    }

    node_index_to_its_order_in_topological_sort_map[p_node->Index()] = i;
  }

  // If boundary op found, create forward op output arg used map.
  if (yield_op_order_in_topological_sort >= 0) {
    for (size_t i = 0; i < node_ids.size(); ++i) {
      const Node* p_node = graph.GetNode(node_ids[i]);
      if (p_node == nullptr /* skip removed nodes*/) {
        continue;
      }

      const Node& node = *p_node;
      bool is_forward_op = IsForwardPassOperator(static_cast<int64_t>(i), yield_op_order_in_topological_sort);
      if (!is_forward_op) {
        continue;
      }

      for (auto& output_arg : node.OutputDefs()) {
        bool used_in_fw = false;
        bool used_in_bw = false;
        for (auto& consumer_node : graph.GetConsumerNodes(output_arg->Name())) {
          auto consumer_node_index_in_topological_order =
              node_index_to_its_order_in_topological_sort_map.at(consumer_node->Index());
          if (IsForwardPassOperator(static_cast<int64_t>(consumer_node_index_in_topological_order),
                                    yield_op_order_in_topological_sort)) {
            used_in_fw = true;
          } else {
            used_in_bw = true;
          }
        }
        fw_op_output_arg_used_map.insert({{output_arg->Name(), std::make_pair(used_in_fw, used_in_bw)}});
      }
    }
  }

  // Return whether boundary op is found or not.
  return yield_op_order_in_topological_sort;
}

Status MemoryAlleviation::SelectRecomputeSubgraph(const Node& node,
                                                  const InlinedVector<size_t>& node_output_index_candidates,
                                                  const ActivationUsedMap& fw_op_output_arg_used_map,
                                                  const InlinedHashMap<NodeIndex, size_t>&
                                                      node_index_to_its_order_in_topological_sort_map,
                                                  InlinedVector<const Node*>& nodes,
                                                  const logging::Logger& logger,
                                                  bool compromise_stashed_activation,
                                                  bool& can_compromise_stashed_activation) const {
  can_compromise_stashed_activation = false;

  LOGS(logger, VERBOSE) << "Enter SelectRecomputeSubgraph for Node " << node.Name() << "(" << node.OpType() << ")";
  nodes.clear();

  std::deque<NodeOutputPort> q;
  for (auto output_index : node_output_index_candidates) {
    q.push_back(NodeOutputPort(&node, static_cast<int>(output_index)));
  }

  bool early_stop = false;
  std::set<NodeOutputPort> visited_output_arg_set;
  std::set<const Node*> visited_node_set;
  bool is_entry_node_output_arg = true;
  while (nodes.size() < MAXIMUM_RECOMPUTE_NODE_COUNT && !q.empty() && !early_stop) {
    // Loop all candidate NodeOutputPort, and find the next layer of input nodes.
    size_t current_queue_size = q.size();
    for (size_t i = 0; i < current_queue_size; ++i) {
      NodeOutputPort p = q.front();
      q.pop_front();
      const Node* curr_node = p.first;

      // Skip if the node output is already visited.
      if (std::find(visited_output_arg_set.begin(), visited_output_arg_set.end(), p) !=
          visited_output_arg_set.end()) {
        continue;
      }

      visited_output_arg_set.insert({p});

      // If the node already visited by from it's other output index, skip it.
      if (visited_node_set.find(curr_node) != visited_node_set.end()) {
        continue;
      }

      visited_node_set.insert(curr_node);

      // Buttom-up search rules.
      // If current op is in allowed list, check its input args, and append the producers' NodeOutputPorts to next_q.
      // If current op is NOT in allowed list:
      // 1). the output does not exist in backward, we cannot find a good solution for so, search terminates.
      // 2). the output is used in backward, we don't need trace back further, continue searching.
      auto op_recompute_config_it = recomputable_op_type_to_input_arg_index_map_.find(curr_node->OpType());
      auto cur_output_arg_name = curr_node->OutputDefs()[p.second]->Name();
      if (is_entry_node_output_arg) {
        // We handle the entry node outputs differently because, we don't want this case falls into and succeed one of
        // the checks in the other branch
        // 1. "op is not in recompute op list, but its output is used in backward"
        // 2. "op is in recompute op list, but its output is used in backward"
        // (either of the above checks is true for entry node outputs)
        if (op_recompute_config_it == recomputable_op_type_to_input_arg_index_map_.end()) {
          early_stop = true;
          LOGS(logger, VERBOSE) << "Entry Node " << curr_node->Name() << "(" << curr_node->OpType() << ") is **NOT** "
                                << "in recompute op list, search terminates.";
          break;
        }
      } else {
        if (op_recompute_config_it == recomputable_op_type_to_input_arg_index_map_.end()) {
          if (fw_op_output_arg_used_map.at(cur_output_arg_name).second) {
            LOGS(logger, VERBOSE) << "Node " << curr_node->Name() << "(" << curr_node->OpType() << ") is **NOT** in "
                                  << "recompute op list, but its output [" << cur_output_arg_name
                                  << "] is used in backward, we don't need trace buttom-up further";
            continue;
          } else {
            early_stop = true;
            LOGS(logger, VERBOSE) << "Node " << curr_node->Name() << "(" << curr_node->OpType() << ") is **NOT** in "
                                  << "recompute op list, and its output [" << cur_output_arg_name
                                  << "] does not exist in backward, search terminates.";
            break;
          }
        }

        if (fw_op_output_arg_used_map.at(cur_output_arg_name).second) {
          LOGS(logger, VERBOSE) << "Node " << curr_node->Name() << "(" << curr_node->OpType() << ") "
                                << "is in recompute op list, while its output [" << cur_output_arg_name
                                << "] is used in backward, we don't need trace buttom-up further";
          continue;
        }
      }

      // Append node to the selected graph.
      if (std::find(nodes.begin(), nodes.end(), curr_node) == nodes.end()) {
        nodes.push_back(curr_node);
        LOGS(logger, VERBOSE) << "Node " << curr_node->Name() << "(" << curr_node->OpType()
                              << ") is added in selected subgraph  ";
      }

      float ratio = InputOutputSizeRatio(curr_node);
      float is_current_node_compromisable = (ratio < 1.f);
      can_compromise_stashed_activation = can_compromise_stashed_activation || is_current_node_compromisable;
      if (is_current_node_compromisable) {
        LOGS(logger, VERBOSE) << "Node " << curr_node->Name() << "(" << curr_node->OpType()
                              << ") has input/output size " << ratio << " < 1.f, can compromise stashed activation";
      }

      if (is_current_node_compromisable && compromise_stashed_activation) {
        LOGS(logger, VERBOSE) << "Node " << curr_node->Name() << "(" << curr_node->OpType() << ") is in "
                              << "recompute op list, and its output [" << cur_output_arg_name
                              << "] does not exist in backward, while it meet compromised check, we don't need trace buttom-up further.";
        continue;
      }

      // Iterate all input nodes according to allowed input arg index of the entry node.
      const auto& input_arg_indices = op_recompute_config_it->second.input_arg_indices;
      for (auto it = curr_node->InputEdgesBegin(), end = curr_node->InputEdgesEnd(); it != end; ++it) {
        const Node::EdgeEnd& input_edge = *it;
        const auto& parent_node = input_edge.GetNode();
        const auto parent_node_output_index = input_edge.GetSrcArgIndex();
        const auto current_node_input_index = input_edge.GetDstArgIndex();
        if (std::find(input_arg_indices.begin(), input_arg_indices.end(), current_node_input_index) !=
            input_arg_indices.end()) {
          NodeOutputPort next_p = std::make_pair(&parent_node, parent_node_output_index);

          LOGS(logger, VERBOSE) << "Node " << parent_node.Name() << "(" << parent_node.OpType() << ")'s "
                                << parent_node_output_index
                                << "th output [" << parent_node.OutputDefs()[parent_node_output_index]->Name()
                                << "] is added in recompute search list  ";

          q.push_back(next_p);
        }
      }
    }
    // After handle all entry node outputs, we set the flag to false.
    is_entry_node_output_arg = false;
  }

  // If input args are not found in bw, but op count exceed MAXIMUM_RECOMPUTE_NODE_COUNT, skip recompute.
  if (!q.empty() || early_stop) {
    LOGS(logger, VERBOSE) << "Fail to find a solution for recompute: current node count is " << nodes.size()
                          << ", queue size: " << q.size() << ", early stop: " << early_stop;
    nodes.clear();
  } else {
    // Re-order the nodes in topological order.
    std::sort(nodes.begin(), nodes.end(),
              [&node_index_to_its_order_in_topological_sort_map](const Node*& lhs, const Node*& rhs) {
                return node_index_to_its_order_in_topological_sort_map.at(lhs->Index()) <
                       node_index_to_its_order_in_topological_sort_map.at(rhs->Index());
              });
  }
  return Status::OK();
}

bool MemoryAlleviation::IsNodeRecomputable(const Node& node,
                                           const ActivationUsedMap& fw_op_output_arg_used_map,
                                           const InlinedHashMap<NodeIndex, size_t>&
                                               node_index_to_its_order_in_topological_sort_map,
                                           const InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                               candidate_output_args_map,
                                           AlleviationSubGraphStores& subgraph_stores,
                                           const logging::Logger& logger,
                                           bool compromise_stashed_activation,
                                           bool& can_compromise_stashed_activation) const {
  InlinedVector<const Node*> nodes_in_topological_order;
  ORT_ENFORCE(SelectRecomputeSubgraph(node, candidate_output_args_map.at(&node),
                                      fw_op_output_arg_used_map,
                                      node_index_to_its_order_in_topological_sort_map,
                                      nodes_in_topological_order, logger,
                                      compromise_stashed_activation,
                                      can_compromise_stashed_activation)
                  .IsOK());
  if (nodes_in_topological_order.size() == 0) {
    return false;
  }

  std::string subgraph_str_representation, log_info;
  NodesInTypoOrderToString(nodes_in_topological_order, subgraph_str_representation, log_info);
  LOGS(logger, VERBOSE) << "Node " << node.Name() << "(" << node.OpType() << ") can be recomputed" << log_info;

  // Update the subgraph alleviation config map - key is the subgraph string representation, value is user config.
  UserAlleviationConfig alleviation_config{AlleviationType::None, 0};
  if (pattern_subgraph_to_user_alleviation_config_map_.find(subgraph_str_representation) !=
      pattern_subgraph_to_user_alleviation_config_map_.end()) {
    alleviation_config = pattern_subgraph_to_user_alleviation_config_map_.at(subgraph_str_representation);
  }

  AlleviationSubGraphDesc& subgraph_desc =
      subgraph_stores.Contains(subgraph_str_representation)
          ? subgraph_stores.GetSubGraphDesc(subgraph_str_representation)
          : subgraph_stores.CreateSubGraphDesc(subgraph_str_representation, alleviation_config);

  subgraph_desc.total_frequency += 1;

  // Update the subgraph frequency map - key is the subgraph string representation, value is number of appearances.
  for (size_t output_index : candidate_output_args_map.at(&node)) {
    auto shape_str = TensorShapeProtoToString(node.OutputDefs()[output_index]->Shape());
    subgraph_desc.shape_str_frequency[shape_str]++;
  }

  subgraph_stores.AddRecomputeSubGraphInstance(&node, nodes_in_topological_order, subgraph_desc);

  return true;
}

Status MemoryAlleviation::GetStashedActivationCandidates(const Graph& graph,
                                                         const InlinedHashMap<std::string, std::pair<bool, bool>>&
                                                             fw_op_output_arg_used_map,
                                                         InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                                             candidate_output_args_map,
                                                         const logging::Logger& logger) const {
  for (auto& kv : fw_op_output_arg_used_map) {
    // used by fw and bw, then it is a candidates.
    if (kv.second.first && kv.second.second) {
      const Node* n = graph.GetProducerNode(kv.first);
      ORT_ENFORCE(n, "Activation should have a producer node");
      size_t k = 0;
      for (k = 0; k < n->OutputDefs().size(); ++k) {
        if (n->OutputDefs()[k]->Name().compare(kv.first) == 0) {
          break;
        }
      }

      candidate_output_args_map[n].push_back(k);
      LOGS(logger, VERBOSE) << "Find candidate output named [" << kv.first << "] of Node " << n->Name() << "("
                            << n->OpType() << ")";
    }
  }

  return Status::OK();
}

Status MemoryAlleviation::CreateRecomputeGraph(Graph& graph,
                                               const InlinedVector<const Node*>& nodes_in_topological_order,
                                               Node*& recompute_subgraph_output_node) const {
  InlinedHashMap<NodeArg*, NodeArg*> self_contained_outputs_map;
  for (size_t i = 0; i < nodes_in_topological_order.size(); ++i) {
    Node* node_to_duplicate = graph.GetNode(nodes_in_topological_order[i]->Index());

    // Check whether the node has been recomputed or not. Simply check the existence of the first output
    // of the node has its corresponding recompute name or not.
    if (graph.GetNodeArg(graph_utils::RecomputeName(node_to_duplicate->MutableOutputDefs()[0]->Name())) != nullptr) {
      continue;
    }

    InlinedVector<NodeArg*> new_input_args;
    new_input_args.reserve(node_to_duplicate->MutableInputDefs().size());
    for (NodeArg* input_arg : node_to_duplicate->MutableInputDefs()) {
      if (self_contained_outputs_map.find(input_arg) == self_contained_outputs_map.end()) {
        NodeArg* recompute_input_arg = graph.GetNodeArg(graph_utils::RecomputeName(input_arg->Name()));
        new_input_args.push_back(recompute_input_arg ? recompute_input_arg : input_arg);
      } else {
        new_input_args.push_back(self_contained_outputs_map[input_arg]);
      }
    }

    InlinedVector<NodeArg*> new_output_args;
    new_output_args.reserve(node_to_duplicate->MutableOutputDefs().size());
    for (size_t k = 0; k < node_to_duplicate->MutableOutputDefs().size(); ++k) {
      const auto& output = node_to_duplicate->MutableOutputDefs()[k];
      new_output_args.push_back(&graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                          output->TypeAsProto()));

      self_contained_outputs_map[output] = new_output_args.back();
    }

    Node& recompute_node = graph.AddNode(node_to_duplicate->Name() + "_recompute",
                                         node_to_duplicate->OpType(),
                                         "Recompute of " + node_to_duplicate->Name(),
                                         new_input_args,
                                         new_output_args,
                                         &node_to_duplicate->GetAttributes(),
                                         node_to_duplicate->Domain());

    recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
    recompute_node.SetExecutionProviderType(node_to_duplicate->GetExecutionProviderType());
    ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(recompute_node),
                      "Failed to set op schema for added recompute node.");

    recompute_subgraph_output_node = &recompute_node;

    for (size_t j = 0; j < recompute_node.MutableOutputDefs().size(); ++j) {
      graph.UpdateProducerNode(recompute_node.MutableOutputDefs()[j]->Name(), recompute_node.Index());
    }

    // Add the edges from the recompute node to the original node.
    for (size_t j = 0; j < recompute_node.MutableInputDefs().size(); ++j) {
      NodeArg* input_arg = recompute_node.MutableInputDefs()[j];
      const Node* producer_node = graph.GetProducerNode(input_arg->Name());
      if (producer_node == nullptr) {
        // Skip when it is graph input or initializer.
        continue;
      }
      int producer_output_index = optimizer_utils::IndexOfNodeOutput(*producer_node, *input_arg);
      graph.AddEdge(producer_node->Index(), recompute_node.Index(), static_cast<int>(producer_output_index),
                    static_cast<int>(j));

      graph.AddConsumerNode(input_arg->Name(), &recompute_node);
    }
  }

  return Status::OK();
}

bool MemoryAlleviation::ModifyGraphForRecompute(Graph& graph,
                                                const InlinedHashMap<NodeIndex, size_t>& node_index_to_its_order_in_topological_sort_map,
                                                const InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map,
                                                const logging::Logger& logger,
                                                int64_t boundary_op_order_in_topological_sort,
                                                AlleviationSubGraphStores& subgraph_stores, Node* node) const {
  bool graph_is_modified = false;
  if (subgraph_stores.SubGraphDescCount() == 0) {
    return graph_is_modified;
  }

  AlleviationSubGraphStores::GraphInstanceInfo& sub_graph_instance_info =
      subgraph_stores.GetRecomputeSubGraphInstance(node);

  AlleviationSubGraphDesc& subgraph_desc = subgraph_stores.GetSubGraphDesc(sub_graph_instance_info.second);
  UserAlleviationConfig alleviation_config = subgraph_desc.user_alleviation_config;
  int skip_count = (alleviation_config.requested_count == -1)
                       ? 0
                       : std::max(0, subgraph_desc.total_frequency - alleviation_config.requested_count);

  subgraph_desc.skip_count += 1;

  if (alleviation_config.type != AlleviationType::None && subgraph_desc.skip_count > skip_count) {
    subgraph_desc.applied_count += 1;
    Node* replacement_node = nullptr;
    LOGS(logger, WARNING) << "[Modify Graph] Node " << node->Name() << "(" << node->OpType() << ") is recomputed";
    ORT_ENFORCE(CreateRecomputeGraph(graph, sub_graph_instance_info.first, replacement_node).IsOK());
    ORT_ENFORCE(replacement_node);

    graph_is_modified = true;

    for (size_t output_index : candidate_output_args_map.at(node)) {
      // Collect output edges (connecting to backward ops), to remove.
      std::vector<graph_utils::GraphEdge> output_edges;
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        size_t src_output_idx = static_cast<size_t>(it->GetSrcArgIndex());
        if (src_output_idx != output_index) {
          continue;
        }

        auto tid = node_index_to_its_order_in_topological_sort_map.find(it->GetNode().Index());
        // It is possible the consumer node is newly added as the recompute node, so we need a check here.
        // For those kind of ops, we can treat them as backward ops.
        if (tid == node_index_to_its_order_in_topological_sort_map.end() ||
            !IsForwardPassOperator(node_index_to_its_order_in_topological_sort_map.at(tid->first),
                                   boundary_op_order_in_topological_sort)) {
          // Remove the edge only connecting to backward op.
          output_edges.push_back(graph_utils::GraphEdge::CreateGraphEdge(*node, *it, false));
        }
      }

      if (!output_edges.empty()) {
        // Remove the output edges of the node first
        graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);

        // Create connections between the replacement node and the outgoing nodes.
        for (const auto& output_edge : output_edges) {
          graph.RemoveConsumerNode(node->MutableOutputDefs()[output_index]->Name(), node);

          // Add new edge connecting the input with the output nodes directly.
          // This also updates the destination node's input node args
          graph.AddEdge(replacement_node->Index(), output_edge.dst_node, static_cast<int>(output_index),
                        output_edge.dst_arg_index);
        }
      }
    }
  }

  return graph_is_modified;
}

Status MemoryAlleviation::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& logger)
    const {
  LOGS(logger, VERBOSE) << "Memory alleviation config: " << memory_alleviation_config_ << ", probe level: "
                        << static_cast<int>(level_);

  InlinedHashMap<std::string, std::pair<bool, bool>> fw_op_output_arg_used_map;
  InlinedHashMap<NodeIndex, size_t> node_index_to_its_order_in_topological_sort_map;
  int64_t boundary_op_order_in_topological_sort = PrepareForTransformation(graph,
                                                                           fw_op_output_arg_used_map,
                                                                           node_index_to_its_order_in_topological_sort_map);
  if (boundary_op_order_in_topological_sort < 0) {
    LOGS(logger, VERBOSE) << "No boundary op found. Skip memory alleviation.";
    return Status::OK();
  }

  InlinedHashMap<const Node*, InlinedVector<size_t>> candidate_output_args_map;
  ORT_RETURN_IF_ERROR(GetStashedActivationCandidates(graph, fw_op_output_arg_used_map, candidate_output_args_map,
                                                     logger));

  AlleviationSubGraphStores recompute_subgraph_stores;
  AlleviationSubGraphStores recompute_with_compromise_subgraph_stores;
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // The first pass - find the candidate subgraphs.
  for (int i = static_cast<int>(node_ids.size()) - 1; i >= 0; --i) {
    Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr) {
      continue;
    }

    if (candidate_output_args_map.find(p_node) == candidate_output_args_map.end()) {
      continue;
    }

    bool can_compromise_stashed_activation = false;
    IsNodeRecomputable(*p_node, fw_op_output_arg_used_map, node_index_to_its_order_in_topological_sort_map,
                       candidate_output_args_map,
                       recompute_subgraph_stores, logger, false,
                       can_compromise_stashed_activation);

    if (can_compromise_stashed_activation) {
      LOGS(logger, VERBOSE) << "Searching Node " << p_node->Name() << "(" << p_node->OpType()
                            << ") for compromised recompute";
      // If the subgraph recompute can save memory by comprising the assumption - recompute graphs' input must exist
      // during backward pass, then we can try to compromise the assumption.
      IsNodeRecomputable(*p_node, fw_op_output_arg_used_map, node_index_to_its_order_in_topological_sort_map,
                         candidate_output_args_map,
                         recompute_with_compromise_subgraph_stores, logger, true,
                         can_compromise_stashed_activation);
    }
  }

  // The second pass - apply the transformation.
  // Iterate through the nodes in reversed topological order and find the subgraph that can be alleviated.
  // The reason we do reversed topological order is that we want the later layers' recompute nodes can be appended
  // earlier than the earlier layers, in this way, the execution order of later layers will be in front of the earlier
  // layers.
  for (int i = static_cast<int>(node_ids.size()) - 1; i >= 0; --i) {
    Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr) {
      continue;
    }

    bool has_been_modified = false;
    if (recompute_subgraph_stores.ContainsRecomputeSubGraphInstance(p_node)) {
      has_been_modified = ModifyGraphForRecompute(graph, node_index_to_its_order_in_topological_sort_map,
                                                  candidate_output_args_map, logger,
                                                  boundary_op_order_in_topological_sort, recompute_subgraph_stores, p_node);
    }

    // If there are other recompute plan for this node, we skip them because the graph is already modified.
    if (!has_been_modified && recompute_with_compromise_subgraph_stores.ContainsRecomputeSubGraphInstance(p_node)) {
      has_been_modified = ModifyGraphForRecompute(graph, node_index_to_its_order_in_topological_sort_map,
                                                  candidate_output_args_map, logger,
                                                  boundary_op_order_in_topological_sort,
                                                  recompute_with_compromise_subgraph_stores, p_node);
    }

    modified = modified || has_been_modified;
  }

  PrintSummary(recompute_subgraph_stores, recompute_with_compromise_subgraph_stores, logger);

  return Status::OK();
}

void MemoryAlleviation::NodesInTypoOrderToString(const InlinedVector<const Node*>& nodes_in_topological_order,
                                                 std::string& subgraph_string_representation,
                                                 std::string& log_info) const {
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

std::string MemoryAlleviation::UserAlleviationConfigToString(const UserAlleviationConfig& config) const {
  std::string type_str;
  switch (config.type) {
    case AlleviationType::None: {
      type_str = "Disabled";
    } break;
    case AlleviationType::Recompute: {
      type_str = "Recompute";
    } break;
    default: {
      type_str = "Unknown";
    } break;
  }
  return type_str;
}

void MemoryAlleviation::PrintSummary(const AlleviationSubGraphStores& recompute_stores,
                                     const AlleviationSubGraphStores& recompute_with_compromise_stores,
                                     const logging::Logger& logger) const {
  if (recompute_stores.subgraph_descs.size() == 0 && recompute_with_compromise_stores.subgraph_descs.size() == 0) {
    return;
  }

  std::ostringstream summary;
  summary << "\nMemoryAlleviation Summary:\n";
  summary << "\tUser config:\n\t" << memory_alleviation_config_ << "\n";
  summary << "\t=================================\n";

  auto print_info_from_stores = [&summary, this](std::string store_name, const AlleviationSubGraphStores& stores) {
    summary << "\t########" << store_name << "########\n";
    for (auto subgraph_it = stores.subgraph_descs.begin(); subgraph_it != stores.subgraph_descs.end();
         ++subgraph_it) {
      std::string freq_info;
      if (subgraph_it->second.user_alleviation_config.type != AlleviationType::None)
        freq_info = " (requested_count=" + std::to_string(subgraph_it->second.user_alleviation_config.requested_count) +
                    ", actual applied_count=" +
                    std::to_string(subgraph_it->second.applied_count) + ")";
      summary << "\tSubgraph: " << subgraph_it->first << "\n"
              << "\t\tAlleviationType: "
              << UserAlleviationConfigToString(subgraph_it->second.user_alleviation_config) << freq_info << "\n"
              << "\t\tPatterns: \n";
      for (auto shape_stat_it = subgraph_it->second.shape_str_frequency.begin(); shape_stat_it != subgraph_it->second.shape_str_frequency.end();
           ++shape_stat_it) {
        summary << "\t\t\tPatternShape:" << shape_stat_it->first << "\tFrequency:" << shape_stat_it->second << "\n";
      }
      summary << "\t--------------------------------\n";
    }
    summary << "\t=================================\n";
  };

  print_info_from_stores("Recompute", recompute_stores);
  print_info_from_stores("RecomputeWithCompromise", recompute_with_compromise_stores);

  LOGS(logger, INFO) << summary.str() << "\n";
}

}  // namespace onnxruntime
