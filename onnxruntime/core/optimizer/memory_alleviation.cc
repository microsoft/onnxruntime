// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "core/optimizer/memory_alleviation.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/random_seed.h"
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
static constexpr int32_t MAXIMUM_RECOMPUTE_COUNT = 3;

Status MemoryAlleviation::PrepareCandidateNodes(
    Graph& graph,
    std::unordered_map<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map,
    std::unordered_map<NodeIndex, bool>& is_forward_op_map) const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  is_forward_op_map.clear();
  std::unordered_map<std::string, std::unordered_set<const Node*>> node_arg_name_to_consumer_node_map;
  std::unordered_map<std::string, const Node*> name_to_node_map;
  bool is_forward = true;
  size_t dropout_node_index = 0;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    Node& node = *graph.GetNode(node_ids[i]);
    if (!node.Name().empty()) {
      name_to_node_map[node.Name()] = &node;
    }

    if (enable_dropout_recompute_ == 1) {
      // We assign seed for all dropout nodes in case duplicated/recomputed node use different seeds with original ones.
      if (node.OpType() == "Dropout" || node.OpType() == "BitmaskDropout") {
        node.AddAttribute("seed", static_cast<int64_t>(dropout_node_index + utils::GetRandomSeed()));
        dropout_node_index++;
      }
    }

    if (is_forward) {
      for (auto& output_arg : node.MutableOutputDefs()) {
        for (auto& consumer_node : graph.GetConsumerNodes(output_arg->Name())) {
          node_arg_name_to_consumer_node_map[output_arg->Name()].insert(consumer_node);
        }
      }
    }

    is_forward_op_map[node_ids[i]] = is_forward;

    if (node.OpType() == "YieldOp") {
      is_forward = false;
    }
  }

  fw_op_output_arg_used_map.clear();
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

Status MemoryAlleviation::CheckRecomputeCondition(
    Graph& graph,
    const Node& node,
    std::vector<const Node*>& nodes,
    const std::unordered_map<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map) const {
  const Node* start_node = &node;
  if (node.OpType() == "Reshape") {
    const Node* prev_node = graph.GetProducerNode(node.InputDefs()[0]->Name());
    start_node = prev_node;
    nodes.push_back(&node);
  }

  bool need_check = recompute_op_type_to_input_arg_index_map_.find(start_node->OpType()) !=
                    recompute_op_type_to_input_arg_index_map_.end();

  if (!need_check) {
    nodes.clear();
    return Status::OK();
  }

  std::deque<std::pair<const Node*, int>> q;
  std::set<std::pair<const Node*, int>> visited_output_args_map;

  /// Start node handling....
  nodes.push_back(start_node);
  const auto& input_arg_indices = recompute_op_type_to_input_arg_index_map_.at(start_node->OpType());
  for (auto it = start_node->InputEdgesBegin(), end = start_node->InputEdgesEnd(); it != end; ++it) {
    const Node::EdgeEnd& input_edge = *it;
    // For start node, we only trace back from the specified output port.
    if (std::find(input_arg_indices.begin(), input_arg_indices.end(), it->GetSrcArgIndex()) !=
        input_arg_indices.end()) {
      std::pair<const Node*, int> p = std::make_pair<const Node*, int>(&(input_edge.GetNode()),
                                                                       input_edge.GetDstArgIndex());
      visited_output_args_map.insert({p});
      q.push_back(p);
    }
  }

  while (nodes.size() < MAXIMUM_RECOMPUTE_COUNT && !q.empty()) {
    std::deque<std::pair<const Node*, int>> next_q;

    // For each node, check it's parent input args.
    while (!q.empty()) {
      std::pair<const Node*, int> p = q.front();
      q.pop_front();

      if (std::find(visited_output_args_map.begin(), visited_output_args_map.end(), p) !=
          visited_output_args_map.end()) {
        continue;
      }

      visited_output_args_map.insert({p});

      const Node* n = p.first;
      // Intermediate node handling....
      for (auto it = n->InputEdgesBegin(), end = n->InputEdgesEnd(); it != end; ++it) {
        // If op is not cheap to recompute, then we stop here.
        if (cheap_to_recompute_op_type_list_.find(it->GetNode().OpType()) == cheap_to_recompute_op_type_list_.end()) {
          continue;
        }

        std::vector<std::pair<const Node*, int>> next_input_args;
        const auto& functor = cheap_to_recompute_op_type_list_.at(n->OpType());
        if (functor(graph, it->GetNode(), next_input_args)) {
          if (std::find(nodes.begin(), nodes.end(), &(it->GetNode())) == nodes.end()) {
            nodes.push_back(&(it->GetNode()));
          }

          // continue track the input args of cheap computed nodes.
          for (auto next_input_arg_it = next_input_args.begin();
               next_input_arg_it != next_input_args.end(); ++next_input_arg_it) {
            // If the input arg of recomputed node not exist in bw, then we need trace back.
            if (!fw_op_output_arg_used_map.at(next_input_arg_it->first->OutputDefs()[next_input_arg_it->second]->Name()).second) {
              next_q.push_back(*next_input_arg_it);
            };
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
    std::rotate(nodes.rbegin(), nodes.rbegin() + 1, nodes.rend());
  }

  return Status::OK();
}

Status MemoryAlleviation::ApplyImpl(
    Graph& graph,
    bool& modified,
    int /*graph_level*/,
    const logging::Logger& /*logger*/) const {
  std::unordered_map<std::string, std::pair<bool, bool>> fw_op_output_arg_used_map;
  std::unordered_map<NodeIndex, bool> is_forward_op_map;
  ORT_RETURN_IF_ERROR(PrepareCandidateNodes(graph, fw_op_output_arg_used_map, is_forward_op_map));

  std::vector<std::string> candidate_node_arg_names;
  std::vector<NodeIndex> candidate_node_ids;
  for (auto& kv : fw_op_output_arg_used_map) {
    // used by fw and bw, then it is a good candidates.
    if (kv.second.first && kv.second.second) {
      const Node* n = graph.GetProducerNode(kv.first);
      candidate_node_arg_names.push_back(kv.first);
      candidate_node_ids.push_back(n->Index());
      std::cout << "Find candidate output named " << kv.first << " of Node " << n->Name() << n->OpType() << std::endl;
    }
  }

  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();
  // Traverse backward from the bottom of the graph, so that the recompute nodes
  // for lower layers are executed earlier
  for (int i = static_cast<int>(node_ids.size() - 1); i >= 0; --i) {
    Node& node = *graph.GetNode(node_ids[i]);

    if (std::find(candidate_node_ids.begin(), candidate_node_ids.end(), node.Index()) == candidate_node_ids.end()) {
      continue;
    }

    Node* replacement_node = nullptr;
    std::vector<size_t> output_indices_to_replace;
    for (size_t k = 0; k < node.MutableOutputDefs().size(); ++k) {
      auto output_arg = node.MutableOutputDefs()[k];
      if (std::find(candidate_node_arg_names.begin(), candidate_node_arg_names.end(), output_arg->Name()) !=
          candidate_node_arg_names.end()) {
        output_indices_to_replace.push_back(k);
      }
    }

    std::vector<const Node*> to_duplicated_nodes;
    ORT_RETURN_IF_ERROR(CheckRecomputeCondition(graph, node, to_duplicated_nodes, fw_op_output_arg_used_map));
    if (to_duplicated_nodes.size() > 0) {
      std::unordered_map<NodeArg*, NodeArg*> self_contained_outputs_map;

      for (const Node* to_duplicated_node : to_duplicated_nodes) {
        Node* node_to_replicated = const_cast<Node*>(to_duplicated_node);
        std::vector<NodeArg*> new_input_args;
        new_input_args.reserve(node_to_replicated->MutableInputDefs().size());
        for (NodeArg* input_arg : node_to_replicated->MutableInputDefs()) {
          new_input_args.push_back(self_contained_outputs_map.find(input_arg) == self_contained_outputs_map.end()
                                       ? input_arg
                                       : self_contained_outputs_map[input_arg]);
        }

        std::vector<NodeArg*> new_output_args;
        new_output_args.reserve(node_to_replicated->MutableOutputDefs().size());
        for (size_t k = 0; k < node_to_replicated->MutableOutputDefs().size(); ++k) {
          const auto& output = node_to_replicated->MutableOutputDefs()[k];
          new_output_args.push_back(&graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                              output->TypeAsProto()));

          self_contained_outputs_map[output] = new_output_args.back();
        }

        Node& recompute_node = graph.AddNode(node_to_replicated->Name() + "_recompute",
                                             node_to_replicated->OpType(),
                                             "Recompute of " + node_to_replicated->Name(),
                                             new_input_args,
                                             new_output_args,
                                             &node_to_replicated->GetAttributes(),
                                             node_to_replicated->Domain());

        recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
        recompute_node.SetExecutionProviderType(node_to_replicated->GetExecutionProviderType());

        replacement_node = &recompute_node;

        std::cout << "Node " << node.Name() << " is recomputed" << std::endl;
      }

      if (replacement_node) {
        for (size_t output_index : output_indices_to_replace) {
          std::vector<graph_utils::GraphEdge> output_edges;  //= GraphEdge::GetNodeOutputEdges(node, output_idx);
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

            // Create connections between the replacement node and the outgoing nodes
            for (const auto& output_edge : output_edges) {
              // Add new edge connecting the input with the output nodes directly.
              // This also updates the destination node's input node args
              graph.AddEdge(replacement_node->Index(), output_edge.dst_node, output_index, output_edge.dst_arg_index);
            }
          }
        }

        modified = true;
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
