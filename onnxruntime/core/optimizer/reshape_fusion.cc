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

    if (ReshapeFusion::Fuse_Subgraph1(reshape, graph, logger) ||
        ReshapeFusion::Fuse_Subgraph2(reshape, graph, logger)) {
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
             /            \
         Shape            Shape
            |              |                   (one or two int64[] constant initializers)
         Gather(indice=0)  Gather(indice=1)    a[]        b[] (optional)
            \              /                   /          /
        Unsqueeze      Unsqueeze              /          /
              \        /  ___________________/          /
[input node]   \      /  / ____________________________/
     \          \    /  / /
      \           Concat
       \         /
         Reshape

After fusion:
    [Sub-graph Root Node]   (Constant Initializers, b is optional)
                  \         [0, 0, a, b]
                   \        /
                    Reshape
*/
bool ReshapeFusion::Fuse_Subgraph1(Node& reshape, Graph& graph, const logging::Logger& logger) {
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

  if(!Replace_Node_With_Initializer(graph, concat, shape_value, logger)) {
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

  Remove_Unused_nodes(graph, nodes_to_remove);

  return true;
}

void ReshapeFusion::Remove_Unused_nodes(Graph& graph, const std::vector<Node*> nodes_to_remove) {
  for (Node* node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }
}

bool ReshapeFusion::Replace_Node_With_Initializer(Graph& graph,
                                                  const Node& node_to_replace,
                                                  const std::vector<int64_t> new_shape,
                                                  const logging::Logger& logger) {
  // Create an initializer with the same name as the node_to_replace node output, and replace the node_to_replace node
  const auto& new_initializer_name = node_to_replace.OutputDefs()[0]->Name();
  if (!graph_utils::CanReplaceNodeWithInitializer(graph, node_to_replace, new_initializer_name, logger)) {
    LOGS(logger, WARNING) << "[Reshape Fusion]Cannot replace node with initializer:" << new_initializer_name;
    return false;
  }

  const auto* shape_def = node_to_replace.OutputDefs()[0];
  ONNX_NAMESPACE::TensorProto shape_initializer_proto;
  shape_initializer_proto.set_name(shape_def->Name());
  shape_initializer_proto.add_dims(static_cast<int64_t>(new_shape.size()));
  shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_initializer_proto.set_raw_data(new_shape.data(), new_shape.size() * sizeof(int64_t));
  auto& new_node_arg = graph_utils::AddInitializer(graph, shape_initializer_proto);
  if (!graph_utils::ReplaceNodeWithInitializer(graph, *graph.GetNode(node_to_replace.Index()), new_node_arg)) {
    return false;
  }
  return true;
}

/**
Apply Reshape Fusion. The fowllowing are subgraphs before and after fusion:

Before fusion:
                                      [Sub-graph Root  Node ]
                                                 |
                                               Shape
                                                 |
                                          Gather(indice=n)
                                                /
(one int64[] constant initializer)         Unsqueeze
            a[]                               /
              \           ___________________/
               \         /
[input node]    \       /  
      \           Concat
       \         /
         Reshape

After fusion:
    [Sub-graph Root Node]   (Constant Initializer)
                  \         [a, nth element of Shape node]
                   \        /
                    Reshape
*/
bool ReshapeFusion::Fuse_Subgraph2(Node& reshape, Graph& graph, const logging::Logger& logger) {
  const Node* p_concat = graph_utils::GetInputNode(reshape, 1);
  if (nullptr == p_concat) {
    return false;
  }
  const Node& concat = *p_concat;

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(concat, "Concat", {1, 4, 11})) {
    return false;
  }

  auto concat_input_count = concat.InputArgCount().front();
  if (concat_input_count != 2 || concat.GetOutputEdgesCount() > 1) {
    return false;
  }

  // Compose the shape value input for reshape op.
  std::vector<int64_t> shape_value = {};
  // We do not check whether the initializer is constant.
  // Some model uses constant initializer and some does not.
  // Here we assume that no one will override the initializer using graph input.
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[0]), shape_value)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> parent_path {
      {0, 1, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Gather", {1, 11}, kOnnxDomain},
      {0, 0, "Shape", {1}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(concat, true, parent_path, edges, logger)) {
    return false;
  }

  const Node& unsqueeze_node = edges[0]->GetNode();
  const Node& gather_node = edges[1]->GetNode();
  const Node& shape_node = edges[2]->GetNode();
  if (unsqueeze_node.GetOutputEdgesCount() != 1 || gather_node.GetOutputEdgesCount() != 1 || shape_node.GetOutputEdgesCount() != 1) {
    return false;
  }

  std::vector<int64_t> axes;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze_node, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(gather_node.InputDefs()[1]->Name(), tensor_proto)) {
    return false;
  }
  Initializer init_const{*tensor_proto, graph.ModelPath()};
  int64_t gather_index = init_const.data<int64_t>()[0];
  const auto shape = shape_node.InputDefs()[0]->Shape();

  // Something is wrong, return false and skip fusion.
  if (gather_index >= shape->dim_size() || !shape->dim(gather_index).has_dim_value())
  {
    return false;
  }

  shape_value.push_back(shape->dim(gather_index).dim_value());

  if(!Replace_Node_With_Initializer(graph, concat, shape_value, logger)) {
    return false;
  }

  // Remove nodes not used anymore.
  std::vector<Node*> nodes_to_remove{
      graph.GetNode(unsqueeze_node.Index()),
      graph.GetNode(gather_node.Index()),
      graph.GetNode(shape_node.Index())};

  Remove_Unused_nodes(graph, nodes_to_remove);

  return true;
}
}  // namespace onnxruntime
