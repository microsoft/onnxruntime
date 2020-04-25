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
index corresponding to the index of the argument.)

Before fusion:
   [Sub-graph    Root    Node ]
    |        /                  \
    |    Shape                   Shape
    |       |                      |
    |    Gather(indices=0)  a[]   Gather(indices=2)     b[]
    |       \              /             /             /
    |   Unsqueeze         /        Unsqueeze          /
    |         \          /  ___________/             /
    |          \        /  / _______________________/
    |           \      /  / /
     \            Concat
      \          /
         Reshape

After fusion:
    [Sub-graph Root Node]   (Constant Initializer)
                  \         [0, a, 0, b]
                   \        /
                    Reshape
*/
bool ReshapeFusion::Fuse_Subgraph1(Node& reshape, Graph& graph, const logging::Logger& logger) {
  const Node* p_root = graph_utils::GetInputNode(reshape, 0);

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
  std::set<std::pair<NodeType, NodeIndex>> candidates_for_removal;
  for (int i = 0; i < concat_input_count; ++i) {
    // First check if the i-th argument is an initializer.
    // We do not check whether the initializer is constant.
    // Some model uses constant initializer and some does not.
    // Here we assume that no one will override the initializer using graph input.
    if (optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[i]), shape_value)) {
      continue;
    }

    // Try to find path [Root] --> Shape --> Gather(indices=i) --> Unsqueeze (axes=0) --> Concat [input i]
    std::vector<graph_utils::EdgeEndToMatch> parent_path{
        {0, i, "Unsqueeze", {1, 11}, kOnnxDomain},
        {0, 0, "Gather", {1, 11}, kOnnxDomain},
        {0, 0, "Shape", {1}, kOnnxDomain}};

    std::vector<const Node::EdgeEnd*> edges;
    if (!graph_utils::FindPath(concat, true, parent_path, edges, logger)) {
      return false;
    }

    const Node& unsqueeze = edges[0]->GetNode();
    const Node& gather = edges[1]->GetNode();
    const Node& shape = edges[2]->GetNode();

    if (graph_utils::GetInputNode(shape, 0) != p_root) {
      return false;
    }

    std::vector<int64_t> axes;
    if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
      return false;
    }

    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather.InputDefs()[1]), int64_t(i), false)) {
      return false;
    }

    shape_value.push_back(0);

    candidates_for_removal.insert({NodeType::Unsqueeze, unsqueeze.Index()});
    candidates_for_removal.insert({NodeType::Gather, gather.Index()});
    candidates_for_removal.insert({NodeType::Shape, shape.Index()});
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
  for (const auto& node_type_and_index : candidates_for_removal) {
    Node* node = graph.GetNode(node_type_and_index.second);
    if (node->GetOutputEdgesCount() == 0 && graph.GetNodeOutputsInGraphOutputs(*node).empty()) {
      graph.RemoveNode(node->Index());
    }
  }

  return true;
}

}  // namespace onnxruntime
