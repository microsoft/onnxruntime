// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/parse_past_state_fusion.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::graph_utils;
using namespace ::onnxruntime::utils;

namespace onnxruntime {

static bool OpTypeVersionAndProviderCheck(const Node& node, const char* op_type,
                                          const std::initializer_list<OperatorSetVersion>& versions,
                                          const std::unordered_set<std::string>& compatible_eps) {
  return IsSupportedOptypeVersionAndDomain(node, op_type, versions) &&
         IsSupportedProvider(node, compatible_eps);
}

static std::vector<uint8_t> UnpackInitializer(const TensorProto& init) {
  std::vector<uint8_t> unpacked_bytes;
  UnpackInitializerData(init, unpacked_bytes);
  return unpacked_bytes;
}

/*
Expected If Then branch:

        (tensor sequence) 
             |
             |
          Identity
             |
             |
        (tensor sequence) 
*/

static bool IsExpectedIfThenBranch(const Graph& if_then_subgraph,
                                   const std::unordered_set<std::string>& compatible_eps) {
  GraphViewer graph_viewer(if_then_subgraph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  // The If (Then) subgraph should only contain one node (Identity)
  if (order.size() != 1) {
    return false;
  }

  for (auto index : order) {
    auto* node = if_then_subgraph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      return false;

    // The node should be an Identity node assigned to a compatible EP
    // consuming a tensor sequence input
    if (OpTypeVersionAndProviderCheck(*node, "Identity", {1, 13, 14, 16}, compatible_eps) &&
        HasSequenceType(*node->InputDefs()[0]->TypeAsProto())) {
      return true;
    }
  }

  return false;
}

/*  Expected Loop subgraph:

         (tensor sequence : loop carried dependency) 
              |
              |
         SequenceInsert --------------------------------- past state seed (initialized tensor) 
              |
              |
         (tensor sequence) 
*/
static bool IsLoopedSequenceInsert(const Graph& loop_subgraph,
                                   const std::unordered_set<std::string>& compatible_eps,
                                   /*out*/ TensorProto& past_state_seed,
                                   /*out*/ const TypeProto*& past_state_seed_type_proto) {
  GraphViewer graph_viewer(loop_subgraph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  // The Loop subgraph should only contain one node (SequenceInsert)
  if (order.size() != 1) {
    return false;
  }

  for (auto index : order) {
    auto* node = loop_subgraph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    if (OpTypeVersionAndProviderCheck(*node, "SequenceInsert", {11}, compatible_eps)) {
      // While looking for the initializer, it is enough to search in the current level of the subgraph
      // as that is the case usually.
      const TensorProto* init = nullptr;
      const auto* past_state_seed_node_arg = node->InputDefs()[1];
      if (loop_subgraph.GetInitializedTensor(past_state_seed_node_arg->Name(), init)) {
        // If the past state seed not an initializer, the fusion is off
        // as it complicates things quite a bit but we don't expect it
        // to be a non-initializer.
        past_state_seed = *init;
        past_state_seed_type_proto = past_state_seed_node_arg->TypeAsProto();
        return true;
      }
    }
  }

  return false;
}

/*
Expected If Else branch:

The condition input is expected to be an initializer of condition (true)

iteration count (M)
   |                  SequenceEmpty (loop carried dependency) 
   |                       |
    \                      |           
      - - - - - - - - -   Loop --------------------------------  cond (true) initializer
                           |
                           |
                     (tensor sequence) 
*/

static bool IsExpectedIfElseBranch(const Graph& if_else_subgraph,
                                   const std::unordered_set<std::string>& compatible_eps,
                                   /*out*/ TensorProto& past_state_seed,
                                   /*out*/ const TypeProto*& past_state_seed_type_proto) {
  GraphViewer graph_viewer(if_else_subgraph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  // The If (Else) subgraph should only contain two node (SequenceEmpty and Loop)
  if (order.size() != 2) {
    return false;
  }

  for (auto index : order) {
    auto* node = if_else_subgraph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    if (!OpTypeVersionAndProviderCheck(*node, "SequenceEmpty", {11}, compatible_eps)) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    // The SequenceEmpty node should feed into a Loop
    if (!OpTypeVersionAndProviderCheck(next_node, "Loop", {11, 13, 16}, compatible_eps)) {
      return false;
    }

    const Node& sequence_empty_node = *node;
    const Node& loop_node = *if_else_subgraph.GetNode(next_node.Index());

    // Ensure that the assigned EPs of the SequenceEmpty and Loop nodes are the same
    if (sequence_empty_node.GetExecutionProviderType() != loop_node.GetExecutionProviderType()) {
      return false;
    }

    // The Loop node should only have one output (the tensor sequence itself)
    if (loop_node.OutputDefs().size() != 1) {
      return false;
    }

    auto& loop_subgraph = *loop_node.GetAttributeNameToSubgraphMap().find("body")->second;
    if (!IsLoopedSequenceInsert(loop_subgraph, compatible_eps, past_state_seed, past_state_seed_type_proto)) {
      return false;
    }

    // Ensure that the Loop's condition input is a boolean initializer that is true
    const auto* cond = loop_node.InputDefs()[1];
    const TensorProto* cond_init = if_else_subgraph.GetInitializer(cond->Name(), true);
    if (cond_init != nullptr) {  // If condition input is empty, then it means true implicitly
      bool cond_init_value = static_cast<bool>(UnpackInitializer(*cond_init)[0]);

      // The Loop's condition initializer should be true
      if (!cond_init_value) {
        return false;
      }
    }
  }

  return true;
}

Status ParsePastStateFusion::ApplyImpl(Graph& graph, bool& modified,
                                       int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // SequenceLength node
    if (!OpTypeVersionAndProviderCheck(*node, "SequenceLength", {11}, GetCompatibleExecutionProviders()) ||
        graph.NodeProducesGraphOutput(*node) || (*node).GetOutputEdgesCount() != 1) {
      continue;
    }

    Node& sequence_length_node = *node;

    // Greater node
    auto& greater_node = *graph.GetNode(sequence_length_node.OutputNodesBegin()->Index());

    if (!OpTypeVersionAndProviderCheck(greater_node, "Greater", {9, 13}, GetCompatibleExecutionProviders()) ||
        graph.NodeProducesGraphOutput(greater_node) || greater_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto* greater_B = greater_node.InputDefs()[1];
    const TensorProto* greater_B_init = graph.GetInitializer(greater_B->Name(), true);
    if (greater_B_init == nullptr) {
      continue;
    }

    int64_t greater_B_init_value = reinterpret_cast<int64_t*>(UnpackInitializer(*greater_B_init).data())[0];

    if (greater_B_init_value != 0) {
      continue;
    }

    // If node
    auto& if_node = *graph.GetNode(greater_node.OutputNodesBegin()->Index());

    if (!OpTypeVersionAndProviderCheck(if_node, "If", {11, 13, 16}, GetCompatibleExecutionProviders()) ||
        graph.NodeProducesGraphOutput(if_node)) {
      continue;
    }

    if (sequence_length_node.GetExecutionProviderType() != if_node.GetExecutionProviderType()) {
      continue;
    }

    // There should two output defs and the first one must be a boolean tensor type
    // and the second one should be of tensor sequence type
    const auto& if_output_defs = if_node.MutableOutputDefs();

    // TODO: Handle cases where-in the boolean output might not be present
    // (if we see such models)
    if (if_output_defs.size() != 2) {
      continue;
    }

    if (!HasElementType(*if_output_defs[0]->TypeAsProto()) ||
        if_output_defs[0]->TypeAsProto()->tensor_type().elem_type() !=
            TensorProto::DataType::TensorProto_DataType_BOOL) {
      continue;
    }

    if (!HasSequenceType(*if_output_defs[1]->TypeAsProto())) {
      continue;
    }

    const auto& if_subgraphs = if_node.GetAttributeNameToSubgraphMap();
    if (!IsExpectedIfThenBranch(*if_subgraphs.find("then_branch")->second, GetCompatibleExecutionProviders())) {
      continue;
    }

    TensorProto past_state_seed;
    const TypeProto* past_state_seed_type_proto;

    if (!IsExpectedIfElseBranch(*if_subgraphs.find("else_branch")->second, GetCompatibleExecutionProviders(),
                                past_state_seed, past_state_seed_type_proto)) {
      continue;
    }

    size_t sequence_at_count = 0;
    for (auto if_node_output = if_node.OutputNodesBegin(), end = if_node.OutputNodesEnd(); if_node_output != end; ++if_node_output) {
      if (OpTypeVersionAndProviderCheck(*if_node_output, "SequenceAt", {11}, GetCompatibleExecutionProviders())) {
        ++sequence_at_count;
      }
    }

    bool proceed_with_fusion = true;

    auto if_node_outputs = if_node.GetOutputEdgesCount();

    // Output defs of the fused node will be one boolean plus the number of downstream
    // SequenceAt nodes
    std::vector<NodeArg*> output_defs_of_fused_node(1 + sequence_at_count, nullptr);
    std::vector<Node*> sequence_at_nodes(sequence_at_count, nullptr);

    output_defs_of_fused_node[0] = if_output_defs[0];
    for (auto it = if_node.OutputNodesBegin(), end = if_node.OutputNodesEnd(); it != end; ++it) {
      auto& if_node_output = *graph.GetNode(it->Index());

      if (sequence_length_node.GetExecutionProviderType() != if_node_output.GetExecutionProviderType()) {
        proceed_with_fusion = false;
        break;
      }

      if (if_node_output.OpType() == "SequenceAt") {
        const auto* position = if_node_output.InputDefs()[1];
        const TensorProto* position_init = graph.GetInitializer(position->Name(), true);

        if (position_init == nullptr) {
          proceed_with_fusion = false;
          break;
        }

        int64_t position_init_value = -1;
        if (position_init->data_type() == TensorProto::DataType::TensorProto_DataType_INT64) {
          position_init_value = reinterpret_cast<int64_t*>(UnpackInitializer(*position_init).data())[0];
        } else {  // if it isn't int64, it has to be int32
          position_init_value = reinterpret_cast<int32_t*>(UnpackInitializer(*position_init).data())[0];
        }

        // Can't handle negative indices at this point
        if (position_init_value < 0) {
          proceed_with_fusion = false;
          break;
        }

        // The value exceeds the number of SequenceAt nodes - this points
        // to a problem with the model - abort here
        if (static_cast<size_t>(position_init_value) > sequence_at_count) {
          proceed_with_fusion = false;
          break;
        }

        sequence_at_nodes[position_init_value] = &if_node_output;
        output_defs_of_fused_node[1 + position_init_value] = if_node_output.MutableOutputDefs()[0];
      }
    }

    if (!proceed_with_fusion) {
      continue;
    }

    // If we have reached here, fusion is ON

    // Add the past state seed as an initializer and add a node arg for that
    past_state_seed.clear_name();
    past_state_seed.set_name(sequence_length_node.Name() + "_" +
                             greater_node.Name() + "_" +
                             if_node.Name() + "_" +
                             past_state_seed.name());
    graph.AddInitializedTensor(past_state_seed);

    auto& past_state_node_arg = graph.GetOrCreateNodeArg(past_state_seed.name(),
                                                         past_state_seed_type_proto);

    // Create a fused node for the subgraph
    auto node_name = graph.GenerateNodeName(sequence_length_node.Name() + "_" +
                                            greater_node.Name() + "_" +
                                            if_node.Name());
    Node& past_state_fusion = graph.AddNode(node_name,
                                            "ParsePastState",
                                            node_name,
                                            {sequence_length_node.MutableInputDefs()[0], &past_state_node_arg},
                                            output_defs_of_fused_node,
                                            nullptr,
                                            onnxruntime::kMSDomain);

    // Set the EP of the fused node to be the EP of the SequenceEmpty node
    past_state_fusion.SetExecutionProviderType(sequence_length_node.GetExecutionProviderType());

    // Adjust relationships (edges)
    auto past_state_fusion_idx = past_state_fusion.Index();

    // Re-route tensor sequence input to SequenceLength to the first input of the fused node
    auto& sequence_length_input_edges = GraphEdge::GetNodeInputEdges(sequence_length_node);
    // Check if there is an input edge feeding SequenceLength as sometimes it could
    // be consuming a graph input in which case the input edge count is 0.
    if (sequence_length_input_edges.size() == 1) {
      graph.AddEdge(sequence_length_input_edges[0].src_node,
                    past_state_fusion_idx,
                    sequence_length_input_edges[0].src_arg_index, 0);
    }

    // Re-route bool output of If so that it is now being fed from the fused node
    auto& if_output_edges = GraphEdge::GetNodeOutputEdges(if_node);
    for (auto cur = if_output_edges.cbegin(), end = if_output_edges.cend(); cur != end; ++cur) {
      if (cur->src_arg_index == 0) {  // 0th index of If is the bool output
        graph.AddEdge(past_state_fusion_idx,
                      cur->dst_node,
                      0,  // 0th index of the fused node is the bool output
                      cur->dst_arg_index);
      }
    }

    // Re-route tensor outputs of SequenceAts so that they are now being fed from the fused node
    for (size_t i = 0; i < sequence_at_count; ++i) {
      auto& sequence_at_output_edges = GraphEdge::GetNodeOutputEdges(*sequence_at_nodes[i]);

      for (auto cur = sequence_at_output_edges.cbegin(), end = sequence_at_output_edges.cend(); cur != end; ++cur) {
        graph.AddEdge(past_state_fusion_idx,
                      cur->dst_node,
                      static_cast<int>(i + 1),  // offset source node arg index by 1 because the first output is the bool output
                      cur->dst_arg_index);
      }
    }

    // Remove all unnecessary relationships/nodes prior to the SequenceAts
    auto& sequence_length_output_edges = GraphEdge::GetNodeOutputEdges(sequence_length_node);
    GraphEdge::RemoveGraphEdges(graph, sequence_length_input_edges);
    GraphEdge::RemoveGraphEdges(graph, sequence_length_output_edges);
    graph.RemoveNode(sequence_length_node.Index());

    auto& greater_output_edges = GraphEdge::GetNodeOutputEdges(greater_node);
    GraphEdge::RemoveGraphEdges(graph, greater_output_edges);
    graph.RemoveNode(greater_node.Index());

    GraphEdge::RemoveGraphEdges(graph, if_output_edges);
    graph.RemoveNode(if_node.Index());

    for (size_t i = 0; i < sequence_at_count; ++i) {
      auto& sequence_at_node = *sequence_at_nodes[i];
      auto& sequence_at_output_edges = GraphEdge::GetNodeOutputEdges(sequence_at_node);

      GraphEdge::RemoveGraphEdges(graph, sequence_at_output_edges);
      graph.RemoveNode(sequence_at_node.Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
