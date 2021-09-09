// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/sequence_construction_fusion.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static bool IsSequenceConstruction(Graph& loop_subgraph,
                                   /*out*/ NodeArg*& tensor_to_be_repeated,
                                   /*out*/ bool& is_tensor_an_initializer,
                                   /*out*/ TensorProto& initialized_tensor) {
  bool is_sequence_construction = false;
  tensor_to_be_repeated = nullptr;
  is_tensor_an_initializer = false;

  GraphViewer graph_viewer(loop_subgraph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  // The Loop subgraph should only contain one node (SequenceInsert)
  if (order.size() != 1) {
    return is_sequence_construction;
  }

  for (auto index : order) {
    auto* node = loop_subgraph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "SequenceInsert", {11})) {
      is_sequence_construction = true;
      tensor_to_be_repeated = node->MutableInputDefs()[1];

       auto& node_arg = loop_subgraph.GetOrCreateNodeArg(tensor_to_be_repeated->Name(), tensor_to_be_repeated->TypeAsProto());
      ORT_IGNORE_RETURN_VALUE(node_arg);

      const TensorProto* init = nullptr;
      if (loop_subgraph.GetInitializedTensor(tensor_to_be_repeated->Name(), init)) {
        initialized_tensor = *init;
        is_tensor_an_initializer = true;
      }
    }
  }

  return is_sequence_construction;
}

Status SequenceConstructionFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
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

    // Would be weird if the SequenceEmpty node fed to a graph output directly, handle it anyway
    // If the SequenceEmpty node fed into some other node, the fusion is off
    if (graph.NodeProducesGraphOutput(*node) || node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    if (next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Loop", {1, 11, 13, 16}) ||
        !graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders())) {
      continue;
    }

        Node& sequence_empty_node = *node;
    Node& loop_node = *graph.GetNode(next_node.Index());

    NodeArg* tensor_to_be_repeated = nullptr;
    bool is_tensor_an_initializer = false;
    TensorProto initialized_tensor;

    auto& subgraph = *loop_node.GetAttributeNameToMutableSubgraphMap().find("body")->second;
    if (!IsSequenceConstruction(subgraph, tensor_to_be_repeated,
                                is_tensor_an_initializer,
                                initialized_tensor)) {
      continue;
    }


    const auto* cond = loop_node.MutableInputDefs()[1];
    const TensorProto* cond_init = graph.GetInitializer(cond->Name(), true);
    if (cond_init != nullptr) {
      // TODO : Ensure that the boolean initializer is true
      float f = 1.2f;
      ORT_IGNORE_RETURN_VALUE(f);
    }

    if (is_tensor_an_initializer) {
      graph.AddInitializedTensor(initialized_tensor);
    }

    auto& node_arg = graph.GetOrCreateNodeArg(tensor_to_be_repeated->Name(), tensor_to_be_repeated->TypeAsProto());

    auto node_name = graph.GenerateNodeName(sequence_empty_node.Name() + "_" + loop_node.Name());
    Node& sequence_construct = graph.AddNode(node_name,
                                             "SequenceConstructByTensorRepeats",
                                             node_name,
                                             {&node_arg, loop_node.MutableInputDefs()[0]},
                                             {loop_node.MutableOutputDefs()},
                                             nullptr,
                                             onnxruntime::kMSDomain);
    // Adjust input and output edges

      auto target_idx = sequence_construct.Index();
      auto& loop_input_name = loop_node.MutableInputDefs()[0]->Name();

      auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(loop_node);


      for (auto cur = input_edges.cbegin(), end = input_edges.cend(); cur != end; ++cur) {
        if (cur->arg_name == loop_input_name) {
          auto target_arg_index = graph_utils::GetNodeInputIndexFromInputName(sequence_construct, cur->arg_name);
          graph.AddEdge(cur->src_node, target_idx, cur->src_arg_index, target_arg_index);        
        }
      }

      graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edges);

    auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(loop_node);

      //graph.AddEdge(target_idx, output_edges[0].dst_node, output_edges[0].src_arg_index, output_edges[0].dst_arg_index);

    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);

    // Remove SequenceEmpty and Loop
      output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(sequence_empty_node);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);
      graph.RemoveNode(sequence_empty_node.Index());

         output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(loop_node);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);
         graph.RemoveNode(loop_node.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
