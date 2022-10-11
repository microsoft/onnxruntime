// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/optimizer/memory_alleviation.h"

using onnxruntime::memory_alleviation::ActivationUsedMap;

namespace onnxruntime {

namespace {
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

std::string AlleviationTypeToString(const memory_alleviation::AlleviationType& type) {
  switch (type) {
    case memory_alleviation::AlleviationType::None: {
      return "Disabled";
    } break;
    case memory_alleviation::AlleviationType::Recompute: {
      return "Recompute";
    } break;
    default: {
      return "Unknown";
    } break;
  }
}

}  // namespace

Status MemoryAlleviation::ParseAlleviationConfigFromString(const std::string& enable_memory_alleviation) {
  if (!enable_memory_alleviation.empty()) {
    memory_alleviation_config_ = enable_memory_alleviation;
    const auto alleviation_map = utils::SplitString(enable_memory_alleviation, ",");
    for (const auto& alleviation_config_per_op_str : alleviation_map) {
      const auto alleviation_config_per_op = utils::SplitString(alleviation_config_per_op_str, ":");
      ORT_RETURN_IF_NOT(alleviation_config_per_op.size() == 2,
                        "Alleviation config for each operator should in format of OpType:AlleviationType.");

      const std::string subgraph_string_representation(alleviation_config_per_op[0]);
      int alleviation_type_int = 0;
      const std::string_view& sv = alleviation_config_per_op[1];
      auto result = std::from_chars(sv.data(), sv.data() + sv.size(), alleviation_type_int);
      ORT_RETURN_IF(result.ec == std::errc::invalid_argument, "Fail to convert allocation type from string.");
      ORT_RETURN_IF_NOT(alleviation_type_int < 2 && alleviation_type_int >= 0,
                        "Invalid alleviation type specified for op ", subgraph_string_representation);
      memory_alleviation::AlleviationType alleviation_type =
          static_cast<memory_alleviation::AlleviationType>(alleviation_type_int);

      // At this point, subgraph_string_representation is a pattern graph string representation.
      pattern_subgraph_to_alleviation_type_map_[subgraph_string_representation] = alleviation_type;
    }
  }

  return Status::OK();
}

Status MemoryAlleviation::ParseAlleviationLevelFromString(const std::string& level) {
  if (level.compare("1") == 0) {
    level_ = memory_alleviation::ProbeLevel::Advanced;
  }

  return Status::OK();
}

Status MemoryAlleviation::PrepareForTransformation(const Graph& graph,
                                                   ActivationUsedMap& fw_op_output_arg_used_map,
                                                   InlinedHashMap<NodeIndex, bool>& is_forward_op_map) const {
  fw_op_output_arg_used_map.clear();
  is_forward_op_map.clear();
  InlinedHashMap<std::string, InlinedHashSet<const Node*>> node_arg_name_to_consumer_node_map;

  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  bool is_forward = true;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node& node = *graph.GetNode(node_ids[i]);
    if (is_forward) {
      for (auto& output_arg : node.OutputDefs()) {
        for (auto& consumer_node : graph.GetConsumerNodes(output_arg->Name())) {
          node_arg_name_to_consumer_node_map[output_arg->Name()].insert(consumer_node);
        }
      }
    }

    is_forward_op_map[node.Index()] = is_forward;
    if (node.OpType() == "YieldOp") {
      is_forward = false;
    }
  }

  for (auto& kv : node_arg_name_to_consumer_node_map) {
    for (auto& consumer_node : kv.second) {
      if (is_forward_op_map[consumer_node->Index()]) {
        fw_op_output_arg_used_map[kv.first].first = true;
      } else {
        fw_op_output_arg_used_map[kv.first].second = true;
      }
    }
  }

  return Status::OK();
}

Status MemoryAlleviation::SelectRecomputeSubgraph(const Graph& graph,
                                                  const Node& node,
                                                  const ActivationUsedMap& fw_op_output_arg_used_map,
                                                  InlinedVector<const Node*>& nodes,
                                                  std::ostringstream& oss,
                                                  std::ostringstream& node_type_in_topological_order) const {
  LOGS_DEFAULT(VERBOSE) << "Enter SelectRecomputeSubgraph for Node " << node.Name() << "(" << node.OpType() << ")";

  const Node* start_node = &node;
  // We handle Reshape specifically because it is reusing buffer of input tensor.
  if (node.OpType() == "Reshape") {
    // Be noted, prev_node could be nullptr.
    const Node* prev_node = graph.GetProducerNode(node.InputDefs()[0]->Name());
    start_node = prev_node;
    nodes.push_back(&node);
  }

  if (!start_node || entry_op_type_to_input_arg_index_map_.find(start_node->OpType()) ==
                         entry_op_type_to_input_arg_index_map_.end()) {
    nodes.clear();
    return Status::OK();
  }

  std::deque<memory_alleviation::NodeOutputPort> q;
  std::set<memory_alleviation::NodeOutputPort> visited_output_args_map;

  // Start entry node handling.
  // Iterate all input nodes according to allowed input arg index of the entry node.
  nodes.push_back(start_node);
  LOGS_DEFAULT(VERBOSE) << "start_node: " << start_node->Name() << "(" << start_node->OpType() << ")";

  const auto& entry_operator_config = entry_op_type_to_input_arg_index_map_.at(start_node->OpType());
  const auto& input_arg_indices = entry_operator_config.input_arg_indices;
  for (auto it = start_node->InputEdgesBegin(), end = start_node->InputEdgesEnd(); it != end; ++it) {
    const Node::EdgeEnd& input_edge = *it;
    const auto& parent_node = input_edge.GetNode();
    const auto parent_node_output_index = input_edge.GetSrcArgIndex();
    const auto current_node_input_index = input_edge.GetDstArgIndex();
    // For entry node, we only trace back from the specified output port.
    if (std::find(input_arg_indices.begin(), input_arg_indices.end(), current_node_input_index) !=
        input_arg_indices.end()) {
      memory_alleviation::NodeOutputPort p = std::make_pair(&parent_node, parent_node_output_index);

      LOGS_DEFAULT(VERBOSE) << "Node " << p.first->Name() << "(" << p.first->OpType() << ")'s " << p.second << "th output ["
                            << p.first->OutputDefs()[p.second]->Name() << "] is added in recompute search list  ";

      q.push_back(p);
    }
  }

  bool early_stop = false;
  while (nodes.size() < memory_alleviation::MAXIMUM_RECOMPUTE_NODE_COUNT && !q.empty() && !early_stop) {
    std::deque<memory_alleviation::NodeOutputPort> next_q;

    // Loop all candidate NodeOutputPort, and find the next layer of input nodes.
    while (!q.empty()) {
      memory_alleviation::NodeOutputPort p = q.front();
      q.pop_front();

      if (std::find(visited_output_args_map.begin(), visited_output_args_map.end(), p) !=
          visited_output_args_map.end()) {
        continue;
      }

      visited_output_args_map.insert({p});

      const Node* n = p.first;
      // Intermediate node handling.
      // If op is not in recompute list and
      // 1). the output does not exist in backward, we cannot find a good solution for so, just skip it.
      // 2). the output is used in backward, we don't need trace back further.
      auto entry_operator_config_it = entry_op_type_to_input_arg_index_map_.find(n->OpType());
      auto cur_output_arg_name = n->OutputDefs()[p.second]->Name();
      if (entry_operator_config_it == entry_op_type_to_input_arg_index_map_.end()) {
        if (fw_op_output_arg_used_map.at(cur_output_arg_name).second) {
          LOGS_DEFAULT(VERBOSE) << "Node " << n->Name() << "(" << n->OpType() << ") "
                                << " is **NOT** in recompute op list, but its output " << cur_output_arg_name
                                << "is used in backward, we don't need trace back further";
          continue;
        } else {
          early_stop = true;
          LOGS_DEFAULT(VERBOSE) << "Node " << n->Name() << "(" << n->OpType() << ")'s output " << cur_output_arg_name
                                << " is not in recompute list and the output [" << cur_output_arg_name
                                << "] does not exist in backward, we cannot find a good solution for so, just skip it.";
          break;
        }
      }

      if (fw_op_output_arg_used_map.at(cur_output_arg_name).second) {
        LOGS_DEFAULT(VERBOSE) << "Node " << n->Name() << "(" << n->OpType() << ") "
                              << " is in recompute op list, while its output [" << cur_output_arg_name
                              << "] is used in backward, we don't need trace back further";
        continue;
      }

      // Append node to the selected graph.
      if (std::find(nodes.begin(), nodes.end(), n) == nodes.end()) {
        nodes.push_back(n);
        LOGS_DEFAULT(VERBOSE) << "Node " << n->Name() << "(" << n->OpType() << ") is added in selected subgraph  ";
      }

      const auto& input_arg_indices = entry_operator_config_it->second.input_arg_indices;
      for (auto it = n->InputEdgesBegin(), end = n->InputEdgesEnd(); it != end; ++it) {
        const Node::EdgeEnd& input_edge = *it;
        const auto& parent_node = input_edge.GetNode();
        const auto parent_node_output_index = input_edge.GetSrcArgIndex();
        const auto current_node_input_index = input_edge.GetDstArgIndex();
        // For entry node, we only trace back from the specified output port.
        if (std::find(input_arg_indices.begin(), input_arg_indices.end(), current_node_input_index) !=
            input_arg_indices.end()) {
          memory_alleviation::NodeOutputPort p = std::make_pair(&parent_node, parent_node_output_index);
          LOGS_DEFAULT(VERBOSE) << "Node " << p.first->Name() << "(" << p.first->OpType() << ")'s " << p.second
                                << "th output [" << p.first->OutputDefs()[p.second]->Name()
                                << "] is added in recompute search list";
          next_q.push_back(p);
        }
      }
    }

    q = next_q;
  }

  // If input args are not found in bw, but op count exceed MAXIMUM_RECOMPUTE_NODE_COUNT, skip recompute.
  if (!q.empty() || early_stop) {
    LOGS_DEFAULT(VERBOSE) << "Fail to find a solution for recompute: current node count is " << nodes.size()
                          << ", queue size: " << q.size() << ", early stop: " << early_stop;
    nodes.clear();
  } else {
    std::reverse(nodes.begin(), nodes.end());

    size_t node_count = nodes.size();
    for (size_t i = 0; i < node_count; ++i) {
      if (i < node_count - 1) {  // Ignore the last node.
        oss << "(name:" << nodes[i]->Name() << ", type:" << nodes[i]->OpType() << "),";
      }

      node_type_in_topological_order << nodes[i]->OpType() << "+";
    }
  }
  return Status::OK();
}

Status MemoryAlleviation::GetStashedActivationCandidates(
    const Graph& graph,
    const InlinedHashMap<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map,
    InlinedHashMap<const Node*, InlinedVector<size_t>>& candidate_output_args_map) const {
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
      LOGS_DEFAULT(VERBOSE) << "Find candidate output named " << kv.first << " of Node " << n->Name() << "("
                            << n->OpType() << ")";
    }
  }

  return Status::OK();
}

Status MemoryAlleviation::CreateRecomputeGraph(Graph& graph,
                                               const InlinedVector<const Node*>& nodes_in_topological_order,
                                               Node*& recompute_subgraph_output_node) const {
  const size_t node_count_to_recompute = nodes_in_topological_order.size();
  if (node_count_to_recompute > 0) {
    InlinedHashMap<NodeArg*, NodeArg*> self_contained_outputs_map;
    for (size_t i = 0; i < node_count_to_recompute; ++i) {
      Node* node_to_duplicate = const_cast<Node*>(nodes_in_topological_order[i]);

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
  }

  return Status::OK();
}

Status MemoryAlleviation::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                    const logging::Logger& logger) const {
  InlinedHashMap<std::string, std::pair<bool, bool>> fw_op_output_arg_used_map;
  InlinedHashMap<NodeIndex, bool> is_forward_op_map;
  ORT_RETURN_IF_ERROR(PrepareForTransformation(graph, fw_op_output_arg_used_map, is_forward_op_map));

  InlinedHashMap<const Node*, InlinedVector<size_t>> candidate_output_args_map;
  ORT_RETURN_IF_ERROR(GetStashedActivationCandidates(graph, fw_op_output_arg_used_map, candidate_output_args_map));

  InlinedHashMap<std::string, InlinedHashMap<std::string, int>> stashed_activation_statistics;
  InlinedHashMap<std::string, memory_alleviation::AlleviationType> subgraph_str_to_alleviation_type;
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Iterate through the nodes in reversed topological order and find the subgraph that can be alleviated.
  // The reason we do reversed topological order is that we want the later layers' recompute nodes can be appended
  // earlier than the earlier layers, in this way, the execution order of later layers will be in front of the earlier layers.
  for (int i = static_cast<int>(node_ids.size()) - 1; i >= 0; --i) {
    Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr) {
      continue;
    }

    Node& node = *p_node;
    if (candidate_output_args_map.find(&node) == candidate_output_args_map.end()) {
      continue;
    }

    InlinedVector<const Node*> nodes_in_topological_order;
    std::ostringstream oss;
    std::ostringstream node_type_in_topological_order;
    ORT_RETURN_IF_ERROR(SelectRecomputeSubgraph(graph, node, fw_op_output_arg_used_map, nodes_in_topological_order,
                                                oss, node_type_in_topological_order));
    if (nodes_in_topological_order.size() == 0) {
      continue;
    }

    std::string postfix_if_any;
    if (oss.tellp() != std::streampos(0)) {
      postfix_if_any = " with its precedent nodes: " + oss.str();
    }

    LOGS(logger, VERBOSE) << "Node " << node.Name() << "(" << node.OpType() << ") can be recomputed" << postfix_if_any;

    Node* replacement_node = nullptr;

    auto alleviation_type = memory_alleviation::AlleviationType::None;
    std::string node_type_in_topological_order_str = node_type_in_topological_order.str();
    if (pattern_subgraph_to_alleviation_type_map_.find(node_type_in_topological_order_str) !=
        pattern_subgraph_to_alleviation_type_map_.end()) {
      alleviation_type = pattern_subgraph_to_alleviation_type_map_.at(node_type_in_topological_order_str);
    }
    bool modify_graph = (alleviation_type != memory_alleviation::AlleviationType::None);
    if (modify_graph) {
      ORT_RETURN_IF_ERROR(CreateRecomputeGraph(graph, nodes_in_topological_order, replacement_node));
      ORT_ENFORCE(replacement_node);
    }

    subgraph_str_to_alleviation_type[node_type_in_topological_order_str] = alleviation_type;

    for (size_t output_index : candidate_output_args_map[&node]) {
      if (modify_graph) {
        // Collect output edges (connecting to backward ops), to remove.
        std::vector<graph_utils::GraphEdge> output_edges;
        for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
          size_t src_output_idx = static_cast<size_t>(it->GetSrcArgIndex());
          if (src_output_idx != output_index) {
            continue;
          }

          if (!is_forward_op_map[it->GetNode().Index()]) {
            // Remove the edge only connecting to backward op.
            output_edges.push_back(graph_utils::GraphEdge::CreateGraphEdge(node, *it, false));
          }
        }

        if (!output_edges.empty()) {
          // Remove the output edges of the node first
          graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);

          // Create connections between the replacement node and the outgoing nodes.
          for (const auto& output_edge : output_edges) {
            graph.RemoveConsumerNode(node.MutableOutputDefs()[output_index]->Name(), &node);

            // Add new edge connecting the input with the output nodes directly.
            // This also updates the destination node's input node args
            graph.AddEdge(replacement_node->Index(), output_edge.dst_node, static_cast<int>(output_index),
                          output_edge.dst_arg_index);
          }
        }
      }

      auto shape_str = TensorShapeProtoToString(node.OutputDefs()[output_index]->Shape());
      stashed_activation_statistics[node_type_in_topological_order_str][shape_str]++;
    }

    modified = modify_graph;
  }

  PrintSummary(stashed_activation_statistics, subgraph_str_to_alleviation_type, logger);

  return Status::OK();
}

void MemoryAlleviation::PrintSummary(const InlinedHashMap<std::string, InlinedHashMap<std::string, int>>&
                                         stashed_activation_statistics,
                                     const InlinedHashMap<std::string, memory_alleviation::AlleviationType>&
                                         subgraph_str_to_alleviation_type,
                                     const logging::Logger& logger) const {
  if (stashed_activation_statistics.size() == 0) {
    return;
  }

  std::ostringstream summary;
  summary << "\nMemoryAlleviation Summary:\n";
  summary << "\tUser config:\n\t" << memory_alleviation_config_ << "\n";
  summary << "\t=================================\n";

  for (auto subgraph_it = stashed_activation_statistics.begin(); subgraph_it != stashed_activation_statistics.end();
       ++subgraph_it) {
    summary << "\tSubgraph: " << subgraph_it->first << "\n"
            << "\t\tAlleviationType: " << AlleviationTypeToString(subgraph_str_to_alleviation_type.at(subgraph_it->first)) << "\n"
            << "\t\tPatterns: \n";
    for (auto shape_stat_it = subgraph_it->second.begin(); shape_stat_it != subgraph_it->second.end();
         ++shape_stat_it) {
      summary << "\t\t\tShape:" << shape_stat_it->first << "\n"
              << "\t\t\tFrequency:" << shape_stat_it->second << "\n";
    }
    summary << "\t--------------------------------\n";
  }
  summary << "\t=================================\n";
  LOGS(logger, WARNING) << summary.str() << "\n";
}

}  // namespace onnxruntime
