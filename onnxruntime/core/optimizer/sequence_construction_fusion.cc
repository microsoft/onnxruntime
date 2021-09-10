// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/sequence_construction_fusion.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::graph_utils;

namespace onnxruntime {

static bool IsSequenceConstruction(Graph& loop_subgraph,
                                   /*out*/ NodeArg*& node_arg_of_tensor_to_be_repeated,
                                   /*out*/ TensorProto& tensor_to_be_repeated) {
  node_arg_of_tensor_to_be_repeated = nullptr;

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

    if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "SequenceInsert", {11})) {
      node_arg_of_tensor_to_be_repeated = node->MutableInputDefs()[1];

      // While looking for the initializer, it is enough to search in the current level of the subgraph
      // as that is the case usually. If it were in a different level, the fusion is off as that complicates
      // things quite a bit.
      const TensorProto* init = nullptr;
      if (loop_subgraph.GetInitializedTensor(node_arg_of_tensor_to_be_repeated->Name(), init)) {
        tensor_to_be_repeated = *init;
        return true;
      } else {
        // If the tensor to be repeated in the sequence is not an initializer, the fusion is off
        // as it complicates things quite a bit and we don't expect it to be a non-initializer.
        return false;
      }
    }
  }

  return false;
}

Status SequenceConstructionFusion::ApplyImpl(Graph& graph, bool& modified,
                                             int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "SequenceEmpty", {11}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Would be weird if the SequenceEmpty node fed to a graph output directly,
    // if it did the fusion is off.
    // If the SequenceEmpty node fed into some other node, the fusion is off as well
    if (graph.NodeProducesGraphOutput(*node) || node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    // The SequenceEmpty node should feed into a Loop
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Loop", {1, 11, 13, 16}) ||
        !graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    Node& sequence_empty_node = *node;
    Node& loop_node = *graph.GetNode(next_node.Index());

    // Ensure that the assigned EPs of the SequenceEmpty and Loop nodes are the same
    if (sequence_empty_node.GetExecutionProviderType() != loop_node.GetExecutionProviderType()) {
      continue;
    }

    NodeArg* node_arg_of_tensor_to_be_repeated = nullptr;
    TensorProto tensor_to_be_repeated;

    // The Loop node should only have one output (the tensor sequence itself)
    if (loop_node.MutableOutputDefs().size() != 1) {
      continue;
    }

    auto& loop_subgraph = *loop_node.GetAttributeNameToMutableSubgraphMap().find("body")->second;
    if (!IsSequenceConstruction(loop_subgraph, node_arg_of_tensor_to_be_repeated,
                                tensor_to_be_repeated)) {
      continue;
    }

    // Ensure that the Loop's condition input is a boolean initializer that is true
    const auto* cond = loop_node.MutableInputDefs()[1];
    const TensorProto* cond_init = graph.GetInitializer(cond->Name(), true);
    if (cond_init != nullptr) {  // If condition input is empty, then it means true implicitly
      std::vector<uint8_t> unpacked;
      utils::UnpackInitializerData(*cond_init, unpacked);
      bool cond_init_value = static_cast<bool>(unpacked[0]);

      // The Loop's condition initializer should be true
      if (!cond_init_value) {
        continue;
      }
    }

    // Add the tensor to be repeated which is an initializer in the Loop subgraph
    // to the current subgraph (the Loop node will be removed eventually)
    graph.AddInitializedTensor(tensor_to_be_repeated);

    // Add a node arg corresponding to the newly added initializer to the current graph level
    auto& node_arg = graph.GetOrCreateNodeArg(node_arg_of_tensor_to_be_repeated->Name(),
                                              node_arg_of_tensor_to_be_repeated->TypeAsProto());

    // Create a fused node for the SequenceEmpty + Loop combination
    auto node_name = graph.GenerateNodeName(sequence_empty_node.Name() + "_" + loop_node.Name());
    Node& sequence_construct = graph.AddNode(node_name,
                                             "SequenceConstructionWithTensorAndRepeat",
                                             node_name,
                                             {&node_arg, loop_node.MutableInputDefs()[0]},
                                             {loop_node.MutableOutputDefs()},
                                             nullptr,
                                             onnxruntime::kMSDomain);

    // Set the EP of the fused node to be the EP of the SequenceEmpty node
    sequence_construct.SetExecutionProviderType(sequence_empty_node.GetExecutionProviderType());

    // Adjust input and output edges
    // The following a customized version of `FinalizeNodeFusion()` as logic of moving node inputs
    // and node outputs are a bit more complicated and we can't use the helper directly.

    auto sequence_construct_idx = sequence_construct.Index();

    // Process SequenceEmpty (Remove output edges and the node itself)
    // No input edges to be removed for SequenceEmpty
    auto output_edges = GraphEdge::GetNodeOutputEdges(sequence_empty_node);
    GraphEdge::RemoveGraphEdges(graph, output_edges);
    graph.RemoveNode(sequence_empty_node.Index());

    // Process Loop (Remove input edges, output edges after adjusting the
    // edges to account for the new fused node and then reove the node itself)

    // Process input edges
    auto& loop_iter_count = loop_node.MutableInputDefs()[0]->Name();

    auto loop_input_edges = GraphEdge::GetNodeInputEdges(loop_node);

    for (auto cur = loop_input_edges.cbegin(), end = loop_input_edges.cend(); cur != end; ++cur) {
      // We only care about the edge connection of the loop iteration count
      // Adjust it so that it now feeds the second input of the new fused node
      if (cur->dst_arg_index == 0) {  // iter count is at the 0th node arg index for Loop
        auto target_arg_index = GetNodeInputIndexFromInputName(sequence_construct, cur->arg_name);
        // Adjust it so that it now feeds 1st index of fused node
        graph.AddEdge(cur->src_node, sequence_construct_idx, cur->src_arg_index, 1);
      }
    }

    GraphEdge::RemoveGraphEdges(graph, loop_input_edges);

    // Process output edges
    auto loop_output_edges = GraphEdge::GetNodeOutputEdges(loop_node);

    // If at all the Loop has output edges, it is being fed from its one and only output NodeArg
    // (We have already validated that the Loop has only one NodeArg (the tensor sequence output))
    // Hence, the corresponding source arg index for the fused node will be 0.
    for (auto cur = loop_output_edges.cbegin(), end = loop_output_edges.cend(); cur != end; ++cur) {
      graph.AddEdge(sequence_construct_idx, cur->dst_node, 0, cur->dst_arg_index);
    }

    GraphEdge::RemoveGraphEdges(graph, loop_output_edges);

    // Remove Loop node
    graph.RemoveNode(loop_node.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
