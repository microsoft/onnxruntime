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

static std::vector<uint8_t> UnpackInitializer(const TensorProto& init) {
  std::vector<uint8_t> unpacked_bytes;
  UnpackInitializerData(init, unpacked_bytes);
  return unpacked_bytes;
}

/*
* We want to replace the following pattern:
* 
*  SequenceEmpty   ----------      ----------- other Loop inputs 
*  (loop carried dep)         \   / 
*                             Loop
* 
* 
* with this:
* 
* 
*  Past state seed (initializer)----------                ---- repeat (i.e.) number of elements in the tensor sequence (initializer)
*                                         \             / 
*                                          \           /    
*                                    SequenceConstructUsingTensorAndRepeat 
*                                                |
*                                                |   (loop carried dep)
*                                                |
*                                              Loop --------------------------  other Loop inputs 
* 
* The idea is that the initial "empty" sequence that will get populated on the first Loop iteration
* in the subgraph using the past state seed is directly populated before the Loop is entered. 
* This makes "parsing" the past state within the Loop subgraph much simpler (i.e.) no conditional
* logic is required within the Loop to handle the first iteration and other iterations separately.
*/
static bool AdjustOuterScopeGraph(Graph& parent_graph, const Node& loop, size_t loop_input_index,
                                  TensorProto past_state_seed, const TypeProto*& past_state_seed_type_proto,
                                  int64_t repeat, const std::string& base_name,
                                  const std::unordered_set<std::string>& compatible_eps) {
  Node* node_feeding_loop_input = nullptr;
  for (auto cur = loop.InputEdgesBegin(), end = loop.InputEdgesEnd(); cur != end; ++cur) {
    if (cur->GetDstArgIndex() == static_cast<int>(loop_input_index)) {
      node_feeding_loop_input = parent_graph.GetNode(cur->GetNode().Index());
      break;
    }
  }

  ORT_ENFORCE(node_feeding_loop_input);

  Node& sequence_empty_node = *node_feeding_loop_input;

  // Node feeding loop input must be SequenceEmpty
  if (!IsSupportedOptypeVersionAndDomain(sequence_empty_node, "SequenceEmpty", {11})) {
    return false;
  }

  // Check EP compatibility
  if (!IsSupportedProvider(sequence_empty_node, compatible_eps)) {
    return false;
  }

  // SequenceEmpty should only feed the Loop node
  if (sequence_empty_node.GetOutputEdgesCount() != 1) {
    return false;
  }

  // Add past state seed as an initializer
  past_state_seed.clear_name();
  past_state_seed.set_name(base_name + "_" + "PastStateSeed");
  auto& past_state_seed_node_arg = parent_graph.GetOrCreateNodeArg(past_state_seed.name(),
                                                                   past_state_seed_type_proto);
  parent_graph.AddInitializedTensor(past_state_seed);

  // Add past state repeats as an initializer
  TypeProto repeats_proto;
  repeats_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);

  TensorProto repeats_tensor;
  repeats_tensor.add_int64_data(repeat);
  repeats_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  repeats_tensor.set_name(base_name + "_" + "Repeats");

  auto& past_state_repeats_node_arg = parent_graph.GetOrCreateNodeArg(repeats_tensor.name(), &repeats_proto);
  parent_graph.AddInitializedTensor(repeats_tensor);

  // Add node that will be used instead of SequenceEmpty
  auto node_name = parent_graph.GenerateNodeName(base_name + "_" + "SequenceConstructUsingTensorAndRepeat");
  Node& sequence_construct = parent_graph.AddNode(node_name,
                                                  "SequenceConstructUsingTensorAndRepeat",
                                                  node_name,
                                                  {&past_state_seed_node_arg, &past_state_repeats_node_arg},
                                                  sequence_empty_node.MutableOutputDefs(),
                                                  nullptr,
                                                  onnxruntime::kMSDomain);

  // Set the EP of the fused node to be as the SequenceEmpty node's EP
  sequence_construct.SetExecutionProviderType(sequence_empty_node.GetExecutionProviderType());

  // Adjust the edges
  const auto& sequence_empty_node_output_edges = GraphEdge::GetNodeOutputEdges(sequence_empty_node);

  // The Loop is now to be fed from the fused node
  parent_graph.AddEdge(sequence_construct.Index(),
                       sequence_empty_node_output_edges[0].dst_node,
                       0,
                       sequence_empty_node_output_edges[0].dst_arg_index);

  // Remove SequenceEmpty
  GraphEdge::RemoveGraphEdges(parent_graph, sequence_empty_node_output_edges);
  parent_graph.RemoveNode(sequence_empty_node.Index());

  return true;
}

/* Expected If Then branch: 
   (pass through provided tensor sequence for the non-first iterations of the Loop)

        (tensor sequence) 
             |
             |
          Identity
             |
             |
        (tensor sequence) 
*/

static bool IsExpectedIfThenBranch(const Graph& if_then_subgraph) {
  GraphViewer graph_viewer(if_then_subgraph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  // The If (Then) subgraph should only contain one node (Identity)
  if (order.size() != 1) {
    return false;
  }

  for (auto index : order) {
    auto* node = if_then_subgraph.GetNode(index);
    // check that node hasn't already been removed
    if (!node) {
      return false;
    }

    // The node should be an Identity node assigned to a compatible EP
    // consuming a tensor sequence input
    if (IsSupportedOptypeVersionAndDomain(*node, "Identity", {1, 13, 14, 16}) &&
        HasSequenceType(*node->InputDefs()[0]->TypeAsProto())) {
      return true;
    }
  }

  return false;
}

/*  Expected Loop subgraph for Loop node within If Else branch:
   (create initial past state on the first iteration of the Loop)

         (tensor sequence : loop carried dependency) 
              |
              |
         SequenceInsert --------------------------------- past state seed (initializer) 
              |
              |
         (tensor sequence) 
*/
static bool IsLoopedSequenceInsert(const Graph& loop_subgraph,
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
    if (!node) {
      continue;
    }

    if (!IsSupportedOptypeVersionAndDomain(*node, "SequenceInsert", {11})) {
      continue;
    }

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

  return false;
}

/* 
Expected If Else branch:
(create initial past state on the first iteration of the Loop)

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
    if (!node) {
      continue;
    }

    if (!IsSupportedOptypeVersionAndDomain(*node, "SequenceEmpty", {11})) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    // The SequenceEmpty node should feed into a Loop
    if (!IsSupportedOptypeVersionAndDomain(next_node, "Loop", {11, 13, 16})) {
      return false;
    }

    const Node& loop_node = *if_else_subgraph.GetNode(next_node.Index());

    // The Loop node should only have one output (the tensor sequence itself)
    if (loop_node.OutputDefs().size() != 1) {
      return false;
    }

    auto& loop_subgraph = *loop_node.GetAttributeNameToSubgraphMap().find("body")->second;
    if (!IsLoopedSequenceInsert(loop_subgraph, past_state_seed, past_state_seed_type_proto)) {
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

/*
* We are looking for the following pattern in one level of the graph (possibly the main graph):
* 
*   SequenceEmpty   ----------      ----------- other Loop inputs 
*  (loop carried dep)         \   / 
*                             Loop
* 
* 
* The Loop subgraph should look like this:
* 
* 
*                tensor sequence (loop carried dependency input)
*                     |
*                     |
*                     |
*                SequenceLength
*                     |
*                     |
*                     |
*                 Greater (check if greater than 0)
*                     |
*                     |
*                     |
*                    If (Check the expected then and else subgraphs of If above)
*                   /  \
*                  /    \
*                 /      \
*              bool      tensor sequence (either the input tensor sequence if length > 0 or a tensor sequence filled with a seed tensor) 
*              output      |
*               of         | 
*              Greater     |
*                          |
*                          |
*                          |---------- SequenceAt (0) ----- (tensor)
*                          |
*                          |---------- SequenceAt (1) ----- (tensor)
*                          |
*                          |---------- SequenceAt (n) ----- (tensor)
*                                      (n depends on number of elements in the past state sequence)
* 
* 
* 
* We want to replace the above subgraph within the Loop subgraph into one node:
* 
*                  tensor sequence (loop carried dependency input)
*                               |
*                               |
*                               |
*                     SequenceTensorSplitter
*                          /  |   |   \ 
*         (tensor)    -----   |   |    ----- (tensor)
*                            (tensor)
*
* */
Status ParsePastStateFusion::ApplyImpl(Graph& graph, bool& modified,
                                       int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // We must be within a Loop
    const auto* parent_node = graph.ParentNode();
    if (parent_node == nullptr || parent_node->OpType() != "Loop") {
      continue;
    }

    // SequenceLength node (Must not produce graph output or feed more than one downstream node)
    if (!IsSupportedOptypeVersionAndDomain(*node, "SequenceLength", {11}) ||
        graph.NodeProducesGraphOutput(*node) || (*node).GetOutputEdgesCount() != 1) {
      continue;
    }

    Node& sequence_length_node = *node;

    // Check EP compatibility for SequenceLength
    // We don't need to check EP compatibility for downstream nodes
    // as basically they will all get fused together into one node.
    // We check EP compatibility for this node to ensure that
    // there is a kernel that will be able to run the fused node.
    if (!IsSupportedProvider(sequence_length_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    const auto& sequence_length_node_input = sequence_length_node.InputDefs()[0]->Name();
    int64_t loop_past_state_input_index = -1;
    int64_t iter = 0;
    for (const auto* graph_input : graph.GetInputs()) {
      if (graph_input->Name() == sequence_length_node_input) {
        loop_past_state_input_index = iter;
        break;
      } else {
        ++iter;
      }
    }

    // SequenceLength must be fed by a subgraph input
    if (loop_past_state_input_index == -1) {
      continue;
    }

    // Greater node (Must not produce graph output or feed more than one downstream node)
    auto& greater_node = *graph.GetNode(sequence_length_node.OutputNodesBegin()->Index());

    if (!IsSupportedOptypeVersionAndDomain(greater_node, "Greater", {9, 13}) ||
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

    // If node (Must not produce graph output)
    auto& if_node = *graph.GetNode(greater_node.OutputNodesBegin()->Index());

    if (!IsSupportedOptypeVersionAndDomain(if_node, "If", {11, 13, 16}) ||
        graph.NodeProducesGraphOutput(if_node)) {
      continue;
    }

    // There should two output defs and the first one must be a boolean tensor type
    // and the second one should be of tensor sequence type
    const auto& if_output_defs = if_node.MutableOutputDefs();

    // TODO: Handle cases where-in the boolean output might not be present (if we see such models)
    if (if_output_defs.size() != 2) {
      continue;
    }

    if (!HasElementType(*if_output_defs[0]->TypeAsProto()) ||
        if_output_defs[0]->TypeAsProto()->tensor_type().elem_type() != TensorProto::DataType::TensorProto_DataType_BOOL) {
      continue;
    }

    if (!HasSequenceType(*if_output_defs[1]->TypeAsProto())) {
      continue;
    }

    // Check if the "then" subgraph matches expected pattern
    const auto& if_subgraphs = if_node.GetAttributeNameToSubgraphMap();
    if (!IsExpectedIfThenBranch(*if_subgraphs.find("then_branch")->second)) {
      continue;
    }

    TensorProto past_state_seed;
    const TypeProto* past_state_seed_type_proto;

    // Check if the "else" subgraph matches expected pattern
    if (!IsExpectedIfElseBranch(*if_subgraphs.find("else_branch")->second, past_state_seed, past_state_seed_type_proto)) {
      continue;
    }

    size_t sequence_at_count = 0;
    for (auto if_node_output = if_node.OutputNodesBegin(), end = if_node.OutputNodesEnd();
         if_node_output != end; ++if_node_output) {
      if (IsSupportedOptypeVersionAndDomain(*if_node_output, "SequenceAt", {11})) {
        ++sequence_at_count;
      }
    }

    bool proceed_with_fusion = true;

    // Number of output defs of the fused node will be the number of downstream SequenceAt nodes
    std::vector<NodeArg*> output_defs_of_fused_node(sequence_at_count, nullptr);
    std::vector<Node*> sequence_at_nodes(sequence_at_count, nullptr);

    for (auto it = if_node.OutputNodesBegin(), end = if_node.OutputNodesEnd(); it != end; ++it) {
      auto& if_node_output = *graph.GetNode(it->Index());

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

        sequence_at_nodes[static_cast<size_t>(position_init_value)] = &if_node_output;
        output_defs_of_fused_node[static_cast<size_t>(position_init_value)] = if_node_output.MutableOutputDefs()[0];
      }
    }

    if (!proceed_with_fusion) {
      continue;
    }

    const auto& base_name = sequence_length_node.Name() + "_" +
                            greater_node.Name() + "_" +
                            if_node.Name();

    // Check if the parent graph is conducive for fusion and if so make appropriate
    // parent graph changes before continuing with changes here
    if (!AdjustOuterScopeGraph(*graph.MutableParentGraph(), *parent_node,
                               static_cast<size_t>(loop_past_state_input_index),
                               past_state_seed,
                               past_state_seed_type_proto, sequence_at_count,
                               base_name, GetCompatibleExecutionProviders())) {
      continue;
    }

    // If we have reached here, fusion is ON

    // Create a fused node for the subgraph
    auto node_name = graph.GenerateNodeName(base_name + "_" + "SequenceTensorSplitter");
    Node& sequence_splitter = graph.AddNode(node_name,
                                            "SequenceTensorSplitter",
                                            node_name,
                                            {sequence_length_node.MutableInputDefs()[0]},
                                            output_defs_of_fused_node,
                                            nullptr,
                                            onnxruntime::kMSDomain);

    // Set the EP of the fused node to be the EP of the SequenceEmpty node
    sequence_splitter.SetExecutionProviderType(sequence_length_node.GetExecutionProviderType());

    // Adjust relationships (edges) wrt to the new fused node
    auto sequence_splitter_idx = sequence_splitter.Index();

    // Re-route bool output of If so that it is now being fed from the Greater node
    // after deleting existing output edge from Greater node
    std::vector<NodeIndex> greater_output_nodes;
    std::vector<int> greater_output_dst_args;
    const auto& if_output_edges = GraphEdge::GetNodeOutputEdges(if_node);
    for (auto cur = if_output_edges.cbegin(), end = if_output_edges.cend(); cur != end; ++cur) {
      if (cur->src_arg_index == 0) {  // 0th index of If is the bool output
        greater_output_nodes.push_back(cur->dst_node);
        greater_output_dst_args.push_back(cur->dst_arg_index);
      }
    }

    // Re-route tensor outputs of SequenceAts so that they
    // are now being fed from the SequenceTensorSplitter fused node
    for (size_t i = 0; i < sequence_at_count; ++i) {
      const auto& sequence_at_output_edges = GraphEdge::GetNodeOutputEdges(*sequence_at_nodes[i]);

      for (auto cur = sequence_at_output_edges.cbegin(), end = sequence_at_output_edges.cend();
           cur != end; ++cur) {
        graph.AddEdge(sequence_splitter_idx,
                      cur->dst_node,
                      static_cast<int>(i),
                      cur->dst_arg_index);
      }
    }

    // We are done with SequenceLength - remove it
    GraphEdge::RemoveGraphEdges(graph, GraphEdge::GetNodeOutputEdges(sequence_length_node));
    graph.RemoveNode(sequence_length_node.Index());

    // We are done with If - remove it
    GraphEdge::RemoveGraphEdges(graph, if_output_edges);
    graph.RemoveNode(if_node.Index());

    // Greater node adjustments
    // Input - Adjust the Greater node's input NodeArg so that it is the iter count input of the Loop subgraph
    // The 0th graph input index is the iter count
    greater_node.MutableInputDefs()[0] = &graph.GetOrCreateNodeArg(graph.GetInputs()[0]->Name(), graph.GetInputs()[0]->TypeAsProto());

    // Output - Remove existing edge prior to adding new ones
    GraphEdge::RemoveGraphEdges(graph, GraphEdge::GetNodeOutputEdges(greater_node));
    size_t greater_output_count = greater_output_nodes.size();

    for (size_t i = 0; i < greater_output_count; ++i) {
      graph.AddEdge(greater_node.Index(),
                    greater_output_nodes[i],
                    0,  // 0th index of the Greater node is the bool output
                    greater_output_dst_args[i]);
    }

    // We are done with SequenceAt(s) - remove them
    for (size_t i = 0; i < sequence_at_count; ++i) {
      auto& sequence_at_node = *sequence_at_nodes[i];
      GraphEdge::RemoveGraphEdges(graph, GraphEdge::GetNodeOutputEdges(sequence_at_node));
      graph.RemoveNode(sequence_at_node.Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
