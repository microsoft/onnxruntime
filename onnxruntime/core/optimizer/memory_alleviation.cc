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

namespace memory_alleviation {

AlleviationStratagy ParseFromIntFlag(const int32_t& flag_int) {
  if (flag_int == 0) {
    return AlleviationStratagy::None;
  } else if (flag_int == 1) {
    return AlleviationStratagy::Recompute;
  } else if (flag_int == 2) {
    return AlleviationStratagy::OffloadToCPUMemory;
  } else {
    ORT_THROW("Unknown memory alleviation strategy: ", flag_int);
  }
}
}  // namespace memory_alleviation

namespace {

int32_t SatisfyGeluRecomputeCondition(Graph& /*graph*/,
                                      const Node& node, std::vector<Node*>& nodes,
                                      std::unordered_map<NodeIndex, bool>& /*is_forward_op_map*/) {
  static const InlinedHashSet<std::string_view> target_optypes = {"Gelu", "FastGelu", "BiasGelu"};
  if (target_optypes.find(node.OpType()) == target_optypes.end()) {
    return 0;
  }

  nodes.push_back(const_cast<Node*>(&node));
  return 1;
}

int32_t SatisfyTileRecomputeCondition(Graph& /*graph*/, const Node& node, std::vector<Node*>& nodes,
                                      std::unordered_map<NodeIndex, bool>& /*is_forward_op_map*/) {
  static const InlinedHashSet<std::string_view> target_optypes = {"Tile"};
  if (target_optypes.find(node.OpType()) == target_optypes.end()) {
    return 0;
  }

  nodes.push_back(const_cast<Node*>(&node));
  return 1;
}

// TODO(pengwa): So far this subgraph detection is bound to Ads clrv3 model, we need generalize
// the approach detecting for other  models.
int32_t SatisfyDropoutReshapeMatmulCondition(
    Graph& graph,
    const Node& node,
    std::vector<Node*>& nodes,
    std::unordered_map<NodeIndex, bool>& is_forward_op_map,
    std::unordered_map<std::string, std::pair<bool, bool>>& fw_op_output_arg_used_map) {
  if (node.OpType() != "Reshape") {
    return 0;
  }

  const Node* prev_node = graph.GetProducerNode(node.InputDefs()[0]->Name());
  if (prev_node->OpType() != "Dropout" && prev_node->OpType() != "BitmaskDropout" &&
      !fw_op_output_arg_used_map[prev_node->InputDefs()[0]->Name()].second) {
    return 0;
  }

  const Node* prev_prev_node = graph.GetProducerNode(prev_node->InputDefs()[0]->Name());
  if (prev_prev_node->OpType() != "Where" &&
      !fw_op_output_arg_used_map[prev_prev_node->InputDefs()[0]->Name()].second) {
    return 0;
  }

  // Check input of Dropout is also used by backward.
  std::vector<const Node*> next_nodes;
  next_nodes = graph.GetConsumerNodes(node.OutputDefs()[0]->Name());
  bool find_matmul_in_fw = false;
  bool find_matmul_in_bw = false;
  for (auto next_node : next_nodes) {
    if ((next_node->OpType() == "MatMul" || next_node->OpType() == "FusedMatMul")) {
      if (is_forward_op_map[next_node->Index()]) {
        find_matmul_in_fw = true;
      } else {
        find_matmul_in_bw = true;
      }
    }
  }

  if (find_matmul_in_fw && find_matmul_in_bw) {
    std::cout << "DropoutRecompute: ready to recompute " << node.Name() << node.OpType() << prev_node->OpType()
              << prev_node->Name() << std::endl;
    nodes.push_back(const_cast<Node*>(prev_prev_node));
    nodes.push_back(const_cast<Node*>(prev_node));
    nodes.push_back(const_cast<Node*>(&node));
    return 1;
  }

  return 0;
}

}  // namespace

Status MemoryAlleviation::ApplyImpl(Graph& graph,
                                    bool& modified,
                                    int /*graph_level*/,
                                    const logging::Logger& /*logger*/) const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  std::unordered_map<NodeIndex, bool> is_forward_op_map;
  std::unordered_map<std::string, const Node*> name_to_node_map;
  std::unordered_map<std::string, std::unordered_set<const Node*>> node_arg_name_to_consumer_node_map;

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

  std::unordered_map<std::string, std::pair<bool, bool>> fw_op_output_arg_used_map;
  for (auto& kv : node_arg_name_to_consumer_node_map) {
    for (auto& consumer_node : kv.second) {
      if (is_forward_op_map[consumer_node->Index()]) {
        fw_op_output_arg_used_map[kv.first].first = true;
      } else {
        fw_op_output_arg_used_map[kv.first].second = true;
      }
    }
  }

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
    std::vector<Node*> to_duplicated_nodes;
    int32_t gelu_recompute_flag = 0;
    int32_t dropout_recompute_flag = 0;
    int32_t tile_recompute_flag = 0;
    if ((enable_gelu_recompute_ &&
         (gelu_recompute_flag = SatisfyGeluRecomputeCondition(graph, node, to_duplicated_nodes, is_forward_op_map))) ||
        (enable_dropout_recompute_ &&
         (dropout_recompute_flag = SatisfyDropoutReshapeMatmulCondition(
              graph, node, to_duplicated_nodes, is_forward_op_map, fw_op_output_arg_used_map))) ||
        (enable_tile_recompute_ &&
         (tile_recompute_flag = SatisfyTileRecomputeCondition(graph, node, to_duplicated_nodes, is_forward_op_map)))) {
      std::unordered_map<NodeArg*, NodeArg*> self_contained_outputs_map;

      int32_t action = gelu_recompute_flag * enable_gelu_recompute_ +
                       dropout_recompute_flag * enable_dropout_recompute_ +
                       tile_recompute_flag * enable_tile_recompute_;

      if (action == 1) {  // Recompute nodes
        for (Node* to_duplicated_node : to_duplicated_nodes) {
          std::vector<NodeArg*> new_input_args;
          new_input_args.reserve(to_duplicated_node->MutableInputDefs().size());
          for (NodeArg* input_arg : to_duplicated_node->MutableInputDefs()) {
            new_input_args.push_back(self_contained_outputs_map.find(input_arg) == self_contained_outputs_map.end()
                                         ? input_arg
                                         : self_contained_outputs_map[input_arg]);
          }

          std::vector<NodeArg*> new_output_args;
          new_output_args.reserve(to_duplicated_node->MutableOutputDefs().size());
          for (size_t k = 0; k < to_duplicated_node->MutableOutputDefs().size(); ++k) {
            const auto& output = to_duplicated_node->MutableOutputDefs()[k];
            new_output_args.push_back(&graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                                output->TypeAsProto()));

            self_contained_outputs_map[output] = new_output_args.back();
          }

          Node& recompute_node = graph.AddNode(to_duplicated_node->Name() + "_recompute",
                                               to_duplicated_node->OpType(),
                                               "Recompute of " + to_duplicated_node->Name(),
                                               new_input_args,
                                               new_output_args,
                                               &to_duplicated_node->GetAttributes(),
                                               to_duplicated_node->Domain());

          recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
          recompute_node.SetExecutionProviderType(to_duplicated_node->GetExecutionProviderType());

          replacement_node = &recompute_node;

          std::cout << "Node " << node.Name() << " is recomputed" << std::endl;
        }
      } else if (action == 2) {
        // offload
        Node* last_node = to_duplicated_nodes.back();
        if (last_node->MutableOutputDefs().size() > 1) {
          std::cout << "offload currently only handle the first output when there is multiple outputs, for nodes"
                    << last_node->Name() << "(" << last_node->OpType() << ")" << std::endl;
        }
        std::vector<NodeArg*> new_input_args;
        new_input_args.push_back(last_node->MutableOutputDefs()[0]);

        std::vector<NodeArg*> new_output_args;
        const auto& output = last_node->MutableOutputDefs()[0];
        new_output_args.push_back(&graph.GetOrCreateNodeArg(output->Name() + "_copy_to_host",
                                                            output->TypeAsProto()));

        // KNOWN ISSUE: this offload copy did not help run bigger batch size, need investigate why.
        Node& memcpy_to_host_node = graph.AddNode(last_node->Name() + "_copy_to_host",
                                                  "MemcpyToHost",
                                                  "host copy of " + last_node->Name(),
                                                  new_input_args,
                                                  new_output_args,
                                                  {},
                                                  kOnnxDomain);
        memcpy_to_host_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));
        memcpy_to_host_node.SetExecutionProviderType(last_node->GetExecutionProviderType());

        std::vector<NodeArg*> new_output_args2;
        const auto& output2 = last_node->MutableOutputDefs()[0];
        new_output_args2.push_back(&graph.GetOrCreateNodeArg(output2->Name() + "_copy_to_device",
                                                             output2->TypeAsProto()));

        Node& memcpy_to_device_node = graph.AddNode(last_node->Name() + "_copy_to_device",
                                                    "MemcpyFromHost",
                                                    "device copy of " + last_node->Name(),
                                                    memcpy_to_host_node.MutableOutputDefs(),
                                                    new_output_args2,
                                                    {},
                                                    kOnnxDomain);

        memcpy_to_device_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
        memcpy_to_device_node.SetExecutionProviderType(last_node->GetExecutionProviderType());
        replacement_node = &memcpy_to_device_node;

        std::cout << "Node " << node.Name() << " is offloaded" << std::endl;
      } else {
        ORT_ENFORCE(false, "should not go here.");
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
