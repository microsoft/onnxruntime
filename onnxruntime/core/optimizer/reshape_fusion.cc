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
Apply Reshape Fusion. The fowllowing are subgraphs before and after fusion:

Before fusion:
   [Sub-graph  Root  Node ]
    |        /            \
    |    Shape            Shape
    |       |              |                   (one or two int64[] constant initializers)
    |    Gather(indice=0)  Gather(indice=1)    a[]        b[] (optional)
    |       \              /                   /          /
    |   Unsqueeze      Unsqueeze              /          /
    |         \        /  ___________________/          /
    |          \      /  / ____________________________/
    |           \    /  / /
     \            Concat
      \          /
         Reshape

After fusion:
    [Sub-graph Root Node]   (Constant Initializers, b is optional)
                  \         [0, 0, a, b]
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
  if (concat_input_count < 3 || concat_input_count > 4 || concat.GetOutputEdgesCount() > 1) {
    return false;
  }

  // path 1: [Root] --> Shape --> Gather(indices=0) --> Unsqueeze (axes=0) --> Concat [input 0]
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Gather", {1, 11}, kOnnxDomain},
      {0, 0, "Shape", {1}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(concat, true, parent_path, edges, logger)) {
    return false;
  }

  const Node& unsqueeze_1 = edges[0]->GetNode();
  const Node& gather_1 = edges[1]->GetNode();
  const Node& shape_1 = edges[2]->GetNode();
  if (unsqueeze_1.GetOutputEdgesCount() != 1 || gather_1.GetOutputEdgesCount() != 1 || shape_1.GetOutputEdgesCount() != 1) {
    return false;
  }

  if (graph_utils::GetInputNode(shape_1, 0) != p_root) {
    return false;
  }

  std::vector<int64_t> axes;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze_1, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_1.InputDefs()[1]), int64_t(0), false)) {
    return false;
  }

  // path 2: [Root] --> Shape --> Gather(indices=1) --> Unsqueeze (axes=0) --> Concat [input 1]
  std::vector<graph_utils::EdgeEndToMatch> parent_path2 {
      {0, 1, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Gather", {1, 11}, kOnnxDomain},
      {0, 0, "Shape", {1}, kOnnxDomain}};

  if (!graph_utils::FindPath(concat, true, parent_path2, edges, logger)) {
    return false;
  }

  const Node& unsqueeze_2 = edges[0]->GetNode();
  const Node& gather_2 = edges[1]->GetNode();
  const Node& shape_2 = edges[2]->GetNode();
  if (unsqueeze_2.GetOutputEdgesCount() != 1 || gather_2.GetOutputEdgesCount() != 1 || shape_2.GetOutputEdgesCount() != 1) {
    return false;
  }

  if (graph_utils::GetInputNode(shape_2, 0) != p_root) {
    return false;
  }

  if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze_2, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_2.InputDefs()[1]), int64_t(1), false)) {
    return false;
  }

  // Compose the shape value input for reshape op.
  std::vector<int64_t> shape_value = {0, 0};

  // We do not check whether the initializer is constant.
  // Some model uses constant initializer and some does not.
  // Here we assume that no one will override the initializer using graph input.
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[2]), shape_value)) {
    return false;
  }

  if (concat_input_count > 3) {
    if (!optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[3]), shape_value)) {
      return false;
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

  // Remove nodes not used anymore.
  std::vector<Node*> nodes_to_remove{
      graph.GetNode(unsqueeze_1.Index()),
      graph.GetNode(gather_1.Index()),
      graph.GetNode(shape_1.Index()),
      graph.GetNode(unsqueeze_2.Index()),
      graph.GetNode(gather_2.Index()),
      graph.GetNode(shape_2.Index())};

  for (Node* node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  return true;
}

}  // namespace onnxruntime
