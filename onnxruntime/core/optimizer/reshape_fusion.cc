// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ReshapeFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_reshape = graph.GetNode(node_index);
    if (p_reshape == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reshape = *p_reshape;
    ORT_RETURN_IF_ERROR(Recurse(reshape, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reshape, "Reshape", {5}) ||
        !graph_utils::IsSupportedProvider(reshape, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (ReshapeFusion::Fuse_Subgraph1(reshape, graph, logger)) {
      fused_count++;
      LOGS(logger, INFO) << "Fused reshape node: " << reshape.OutputDefs()[0]->Name();
      modified = true;
    }
  }
  LOGS(logger, INFO) << "Total fused reshape node count: " << fused_count;

  return Status::OK();
}

/**
Apply Reshape Fusion. The following are subgraphs before and after fusion:
(a[] and b[] are int64[] constant initializers; Concat may have any number of arguments,
each of which is a constant initializer or a Shape->Gather->Unsqueeze chain with the
index corresponding to the index of the argument, or a subgraph linked to the subgraph root
and in which nodes have only one input/output.)

Before fusion:
   [Sub-graph    Root]
    |        /                  \
    |    Shape                   Shape
    |       |                      |
    |    Gather(indices=0)  a[]   Gather(indices=2)  b[] or subgraph
    |       \              /             /             /
    |   Unsqueeze         /        Unsqueeze          /
    |         \          /  ___________/             /
    |          \        /  / _______________________/
    |           \      /  / /
     \            Concat
      \          /
         Reshape

After fusion:
    [Sub-graph Root]   (Constant Initializer)
                  \         [0, a, 0, b]
                   \        /
                    Reshape
*/
bool ReshapeFusion::Fuse_Subgraph1(Node& reshape, Graph& graph, const logging::Logger& logger) {
  // The root could be either a graph input or a node so use node arg to compare.
  const NodeArg& root_input = *(reshape.InputDefs()[0]);

  const Node* p_concat = graph_utils::GetInputNode(reshape, 1);
  if (nullptr == p_concat) {
    return false;
  }
  const Node& concat = *p_concat;

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(concat, "Concat", {1, 4, 11})) {
    return false;
  }

  auto concat_input_count = concat.InputArgCount().front();
  if (concat.GetOutputEdgesCount() > 1) {
    return false;
  }

  std::vector<int64_t> shape_value;
  shape_value.reserve(concat_input_count);
  // Used to keep the following nodes in the order of their potential removal.
  enum class NodeType { Unsqueeze, Gather, Shape };
  int subgraph_cnt = 0;
  std::vector<NodeIndex> candidates_for_removal;
  std::vector<NodeIndex> subgraph_candidates;
  for (int i = 0; i < concat_input_count; ++i) {
    // First check if the i-th argument is a constant initializer.
    if (optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[i]), shape_value, true)) {
      continue;
    }

    // Try to find path [Root] --> Shape --> Gather(indices=i) --> Unsqueeze (axes=0) --> Concat [input i]
    std::vector<graph_utils::EdgeEndToMatch> parent_path{
        {0, i, "Unsqueeze", {1, 11}, kOnnxDomain},
        {0, 0, "Gather", {1, 11}, kOnnxDomain},
        {0, 0, "Shape", {1}, kOnnxDomain}};
    std::vector<const Node::EdgeEnd*> edges;
    if (graph_utils::FindPath(concat, true, parent_path, edges, logger)) {
      const Node& unsqueeze = edges[0]->GetNode();
      const Node& gather = edges[1]->GetNode();
      const Node& shape = edges[2]->GetNode();

      const NodeArg& shape_input = *(shape.InputDefs()[0]);
      if (shape_input.Name() != root_input.Name()) {
        return false;
      }

      std::vector<int64_t> axes;
      if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
        return false;
      }

      if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather.InputDefs()[1]), int64_t(shape_value.size()), false)) {
        return false;
      }

      shape_value.push_back(0);

      candidates_for_removal.push_back(unsqueeze.Index());
      candidates_for_removal.push_back(gather.Index());
      candidates_for_removal.push_back(shape.Index()); 
    } else {
      // Find subgraph
      if (subgraph_cnt > 0 || std::find(shape_value.begin(), shape_value.end(), -1) != shape_value.end()) {
        // Only one subgraph can be used to fuse, and each shape node should contain only one value of -1. 
        return false;
      }
      Node* p_cur_node = const_cast<Node*>(graph_utils::GetInputNode(concat, i));
      if (p_cur_node == nullptr) {
        return false;
      }
      subgraph_candidates.clear();
      // From the current node, find the subgraph bottom-up util it reaches the root input node. 
      while (p_cur_node != nullptr) {
        Node& cur_node = *p_cur_node;
        // Each eligible node in the subgraph must have exactly one output node and less than one 
        // input edge(the node input could be graph input). 
        if (cur_node.GetOutputEdgesCount() != 1 || cur_node.GetInputEdgesCount() > 1) {
          break;
        }
        // Match input of current node with root node.
        bool input_matched = false;
        for (size_t j = 0; j < cur_node.InputDefs().size(); j++) {
          if (graph_utils::GetNodeInputName(cur_node, j) == root_input.Name()) {
            input_matched = true;
            break;
          }
        }
        if (!input_matched) {
          // No input in current node matches root input. Go to the next node.
          subgraph_candidates.push_back(cur_node.Index());
          p_cur_node = const_cast<Node*>(graph_utils::GetInputNode(cur_node, 0));
          continue;
        }
        subgraph_cnt++;
        subgraph_candidates.push_back(cur_node.Index());
        shape_value.push_back(-1);
        break;
      }
      if (subgraph_cnt != 1) {
        // No subgraph matched.
        return false;
      }

      for (const auto& nodeIndex : subgraph_candidates){
        candidates_for_removal.push_back(nodeIndex);
      }
    }
  }
  // Create an initializer with the same name as the concat node output, and replace the concat node
  const auto& new_initializer_name = concat.OutputDefs()[0]->Name();
  if (!graph_utils::CanReplaceNodeWithInitializer(graph, concat, new_initializer_name, logger)) {
    LOGS(logger, WARNING) << "Cannot replace concat node with initializer:" << new_initializer_name;
    return false;
  }
  const auto* shape_def = concat.OutputDefs()[0];
  ONNX_NAMESPACE::TensorProto shape_initializer_proto;
  shape_initializer_proto.set_name(shape_def->Name());
  shape_initializer_proto.add_dims(static_cast<int64_t>(shape_value.size()));
  shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_initializer_proto.set_raw_data(shape_value.data(), shape_value.size() * sizeof(int64_t));
  auto& new_node_arg = graph_utils::AddInitializer(graph, shape_initializer_proto);
  if (!graph_utils::ReplaceNodeWithInitializer(graph, *graph.GetNode(concat.Index()), new_node_arg)) {
    return false;
  }

  // Remove nodes that are not used anymore.
  for (const auto& node_index : candidates_for_removal) {
    Node* node = graph.GetNode(node_index);
    if (node->GetOutputEdgesCount() == 0 && graph.GetNodeOutputsInGraphOutputs(*node).empty()) {
      graph.RemoveNode(node->Index());
    }
  }
  return true;
}

}  // namespace onnxruntime
