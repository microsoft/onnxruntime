// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/memory_alleviation.h"
#include "core/optimizer/utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"

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
    const auto alleviation_map = utils::SplitString(enable_memory_alleviation, ",");
    for (const auto& alleviation_config_per_op_str : alleviation_map) {
      const auto alleviation_config_per_op = utils::SplitString(alleviation_config_per_op_str, ":");
      ORT_RETURN_IF_NOT(alleviation_config_per_op.size() == 2,
                        "Alleviation config for each operator should in format of OpType:AlleviationType.");

      const std::string op_type(alleviation_config_per_op[0]);
      int alleviation_type_int = 0;
      const std::string_view& sv = alleviation_config_per_op[1];
      auto result = std::from_chars(sv.data(), sv.data() + sv.size(), alleviation_type_int);
      ORT_RETURN_IF(result.ec == std::errc::invalid_argument, "Fail to convert allocation type from string.");
      ORT_RETURN_IF_NOT(alleviation_type_int < 2 && alleviation_type_int >= 0,
                        "Invalid alleviation type specified for op ", op_type);
      memory_alleviation::AlleviationType alleviation_type =
          static_cast<memory_alleviation::AlleviationType>(alleviation_type_int);

      if (op_type.compare(memory_alleviation::UserConfig_OpTypeGelu) == 0) {
        gelu_alleviation_type_ = alleviation_type;
      } else if (op_type.compare(memory_alleviation::UserConfig_OpTypeTile) == 0) {
        tile_alleviation_type_ = alleviation_type;
      } else if (op_type.compare(memory_alleviation::UserConfig_OpTypeDropout) == 0) {
        dropout_alleviation_type_ = alleviation_type;
      }
    }
  }

  return Status::OK();
}

Status MemoryAlleviation::RegisterRecomputableIntermediateOps() {
  recomputable_intermediate_op_crawler_map_["Where"] =
      [](const Graph& graph, const Node& node,
         InlinedVector<memory_alleviation::NodeOutputPort>& input_node_output_args) -> bool {
    const Node* data_true_node = graph.GetProducerNode(node.InputDefs()[1]->Name());
    if (!data_true_node) {
      // input is graph inptus.
      return true;
    }

    size_t producer_output_index = 0;
    for (size_t i = 0; i < data_true_node->OutputDefs().size(); ++i) {
      if (data_true_node->OutputDefs()[i]->Name().compare(node.InputDefs()[1]->Name()) == 0) {
        producer_output_index = i;
        break;
      }
    }

    // False condition be a scalar constant.
    if (!graph.GetConstantInitializer(node.InputDefs()[2]->Name(), true)) {
      return false;
    }

    input_node_output_args.push_back(
        std::make_pair(std::move(data_true_node), producer_output_index));
    return true;
  };

  return Status::OK();
}

Status MemoryAlleviation::PrepareForTransformation(const Graph& graph,
                                                   ActivationUsedMap& fw_op_output_arg_used_map,
                                                   InlinedHashMap<NodeIndex, bool>& is_forward_op_map) const {
  fw_op_output_arg_used_map.clear();
  is_forward_op_map.clear();
  InlinedHashMap<std::string, std::unordered_set<const Node*>> node_arg_name_to_consumer_node_map;

  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  bool is_forward = true;
  size_t dropout_node_index = 0;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node& node = *graph.GetNode(node_ids[i]);
    if (dropout_alleviation_type_ == memory_alleviation::AlleviationType::Recompute &&
        ((node.OpType() == "Dropout" || node.OpType() == "BitmaskDropout"))) {
      // We assign seed for all dropout nodes in case duplicated/recomputed node use different seeds with original ones.
      const_cast<Node&>(node).AddAttribute("seed", static_cast<int64_t>(dropout_node_index + utils::GetRandomSeed()));
      dropout_node_index++;
    }

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

Status MemoryAlleviation::SelectSubgraph(const Graph& graph,
                                         const Node& node,
                                         const ActivationUsedMap& fw_op_output_arg_used_map,
                                         InlinedVector<const Node*>& nodes,
                                         std::ostringstream& oss,
                                         std::ostringstream& node_type_in_topological_order,
                                         memory_alleviation::AlleviationType& alleviation_type) const {
  LOGS_DEFAULT(VERBOSE) << "Enter SelectSubgraph for Node " << node.Name() << "(" << node.OpType() << ")";

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
  const auto& entry_operator_config = entry_op_type_to_input_arg_index_map_.at(start_node->OpType());
  const auto& input_arg_indices = entry_operator_config.input_arg_indices;
  for (auto it = start_node->InputEdgesBegin(), end = start_node->InputEdgesEnd(); it != end; ++it) {
    const Node::EdgeEnd& input_edge = *it;
    // For entry node, we only trace back from the specified output port.
    if (std::find(input_arg_indices.begin(), input_arg_indices.end(), it->GetSrcArgIndex()) !=
        input_arg_indices.end()) {
      memory_alleviation::NodeOutputPort p = std::make_pair(&(input_edge.GetNode()),
                                                            input_edge.GetDstArgIndex());
      q.push_back(p);
    }
  }

  alleviation_type = entry_operator_config.type;

  while (nodes.size() < memory_alleviation::MAXIMUM_RECOMPUTE_COUNT && !q.empty()) {
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
      // If op is not cheap to recompute, then we stop here.
      // Otherwise, use provided functor to continue next layer of input nodes tracing.
      if (recomputable_intermediate_op_crawler_map_.find(n->OpType()) ==
          recomputable_intermediate_op_crawler_map_.end()) {
        continue;
      }

      InlinedVector<memory_alleviation::NodeOutputPort> input_node_output_args;
      const auto& functor = recomputable_intermediate_op_crawler_map_.at(n->OpType());
      if (functor(graph, *n, input_node_output_args)) {
        if (std::find(nodes.begin(), nodes.end(), n) == nodes.end()) {
          nodes.push_back(n);
        }

        // Continue tracking the producers of input args for the cheap computed node.
        for (auto next_input_arg_it = input_node_output_args.begin();
             next_input_arg_it != input_node_output_args.end(); ++next_input_arg_it) {
          // If the input arg of recomputed node not exist in bw, then we need trace back to see whether that
          // activation can be recomputed or not. Otherwise, we just stop tracing.
          auto& output_arg_name = next_input_arg_it->first->OutputDefs()[next_input_arg_it->second]->Name();
          if (!fw_op_output_arg_used_map.at(output_arg_name).second) {
            next_q.push_back(*next_input_arg_it);
          }
        }
      }
    }

    q = next_q;
  }

  // If input args are not found in bw, but op count exceed MAXIMUM_RECOMPUTE_COUNT, skip recompute.
  if (!q.empty()) {
    nodes.clear();
  } else {
    std::reverse(nodes.begin(), nodes.end());

    size_t node_count = nodes.size();
    for (size_t i = 0; i < node_count; ++i) {
      if (i < node_count - 1) {  // Ignore the last node.
        oss << "(name:" << nodes[i]->Name() << ", type:" << nodes[i]->OpType() << "),";
      }

      node_type_in_topological_order << nodes[i]->OpType() << " + ";
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
      InlinedVector<NodeArg*> new_input_args;
      new_input_args.reserve(node_to_duplicate->MutableInputDefs().size());
      for (NodeArg* input_arg : node_to_duplicate->MutableInputDefs()) {
        new_input_args.push_back(self_contained_outputs_map.find(input_arg) == self_contained_outputs_map.end()
                                     ? input_arg
                                     : self_contained_outputs_map[input_arg]);
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

      recompute_subgraph_output_node = &recompute_node;
    }
  }

  return Status::OK();
}

Status MemoryAlleviation::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                    const logging::Logger& /*logger*/) const {
  InlinedHashMap<std::string, std::pair<bool, bool>> fw_op_output_arg_used_map;
  InlinedHashMap<NodeIndex, bool> is_forward_op_map;
  ORT_RETURN_IF_ERROR(PrepareForTransformation(graph, fw_op_output_arg_used_map, is_forward_op_map));

  InlinedHashMap<const Node*, InlinedVector<size_t>> candidate_output_args_map;
  ORT_RETURN_IF_ERROR(GetStashedActivationCandidates(graph, fw_op_output_arg_used_map, candidate_output_args_map));

  InlinedHashMap<std::string, InlinedHashMap<std::string, int>> stashed_activation_statistics;
  InlinedHashMap<std::string, memory_alleviation::AlleviationType> subgraph_str_to_alleviation_type;
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();
  for (int i = static_cast<int>(node_ids.size() - 1); i >= 0; --i) {
    Node& node = *graph.GetNode(node_ids[i]);
    if (candidate_output_args_map.find(&node) == candidate_output_args_map.end()) {
      continue;
    }

    InlinedVector<const Node*> nodes_in_topological_order;
    std::ostringstream oss;
    std::ostringstream node_type_in_topological_order;
    memory_alleviation::AlleviationType alleviation_type;
    ORT_RETURN_IF_ERROR(SelectSubgraph(graph, node, fw_op_output_arg_used_map, nodes_in_topological_order,
                                       oss, node_type_in_topological_order, alleviation_type));
    if (nodes_in_topological_order.size() == 0) {
      continue;
    }

    std::string postfix_if_any;
    if (oss.tellp() != std::streampos(0)) {
      postfix_if_any = " with its precedent nodes: " + oss.str();
    }
    LOGS_DEFAULT(WARNING) << "Node " << node.Name() << "(" << node.OpType() << ") can be recomputed" << postfix_if_any;

    Node* replacement_node = nullptr;

    bool modify_graph = (alleviation_type != memory_alleviation::AlleviationType::None);

    if (modify_graph) {
      ORT_RETURN_IF_ERROR(CreateRecomputeGraph(graph, nodes_in_topological_order, replacement_node));
      ORT_ENFORCE(replacement_node);
    }

    std::string node_type_in_topological_order_str = node_type_in_topological_order.str();
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
            // Add new edge connecting the input with the output nodes directly.
            // This also updates the destination node's input node args
            graph.AddEdge(replacement_node->Index(), output_edge.dst_node, output_index, output_edge.dst_arg_index);
          }
        }
      }

      auto shape_str = TensorShapeProtoToString(node.OutputDefs()[output_index]->Shape());
      stashed_activation_statistics[node_type_in_topological_order_str][shape_str]++;
    }

    modified = modify_graph;
  }

  PrintSummary(stashed_activation_statistics, subgraph_str_to_alleviation_type);

  return Status::OK();
}

void MemoryAlleviation::PrintSummary(const InlinedHashMap<std::string, InlinedHashMap<std::string, int>>&
                                         stashed_activation_statistics,
                                     const InlinedHashMap<std::string, memory_alleviation::AlleviationType>&
                                         subgraph_str_to_alleviation_type) const {
  if (stashed_activation_statistics.size() == 0) {
    return;
  }

  std::ostringstream summary;
  summary << "\nMemoryAlleviation Summary:\n";
  summary << "\tType config:\n\t\tDropout-" << dropout_alleviation_type_
          << ", Gelu-" << gelu_alleviation_type_ << ", Tile-" << tile_alleviation_type_ << "\n";
  summary << "\t=================================\n";

  for (auto subgraph_it = stashed_activation_statistics.begin(); subgraph_it != stashed_activation_statistics.end();
       ++subgraph_it) {
    summary << "\tSubgraph: " << subgraph_it->first << "\t"
            << AlleviationTypeToString(subgraph_str_to_alleviation_type.at(subgraph_it->first)) << "\n";
    summary << "\t\tShape\tFrequency\n";
    for (auto shape_stat_it = subgraph_it->second.begin(); shape_stat_it != subgraph_it->second.end();
         ++shape_stat_it) {
      summary << "\t\t" << shape_stat_it->first << "\t" << shape_stat_it->second << "\n";
    }
    summary << "\t--------------------------------\n";
  }
  summary << "\t=================================\n";
  LOGS_DEFAULT(WARNING) << summary.str();
}

}  // namespace onnxruntime
