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
 * Find the subgraph that matches [root] -> Shape -> Gather -> Unsqueeze
 */
bool ReshapeFusion::Fuse_Subgraph2(Graph& graph, const NodeArg& root_input, const Node& concat, 
  int index, std::vector<int64_t> shape_value, const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
        {0, index, "Unsqueeze", {1, 11}, kOnnxDomain},
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
    return true;
  }

  return false;
}
/**
Apply Reshape Fusion. The following are subgraphs before and after fusion:
(a[] and b[] are int64[] constant initializers; Concat may have any number of arguments,
each of which is a constant initializer or a Shape->Gather->Unsqueeze chain with the
index corresponding to the index of the argument, or a custom subgraph in which nodes 
have only one output edge. Note the resulting shape value should contain no more than one
value of -1.

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
  if (!optimizer_utils::CheckOutputEdges(graph, concat, 1)) {
    return false;
  }

  // Loop through the inputs of concat node to calculate the shape_value for a potential reshape fusion. 
  std::vector<int64_t> shape_value;
  shape_value.reserve(concat_input_count);
  for (int i = 0; i < concat_input_count; ++i) {
    // First check if the i-th argument is a constant initializer.
    if (optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[i]), shape_value, true)) {
      continue;
    }
    // Try to find path [Root] --> Shape --> Gather(indices=i) --> Unsqueeze (axes=0) --> Concat [input i]
    bool matched = ReshapeFusion::Fuse_Subgraph2(graph, root_input, concat, i, shape_value, logger);
    const Node* p_cur_node = graph_utils::GetInputNode(concat, i);
    if (matched) {
      shape_value.push_back(0);
    } else if (p_cur_node != nullptr) {
      // This node could lead to a potential subgraph pattern fusion. Mark the shape value to -1.
      shape_value.push_back(-1);
    } else {
      return false;
    }
  }  

  // Check how many -1 are there in shape_value.
  int subgraph_cnt = 0;
  for (auto it = shape_value.begin(); it < shape_value.end(); ++it) {
    if ((*it) == -1) {
      subgraph_cnt++;
    }
  }
  // If more than one "-1" value is present in shape_value, return false to exit current fusion.  
  if (subgraph_cnt > 1) {
    return false;
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

  // Safely remove concat parent nodes which have only one output
  for (int i = 0; i < concat_input_count; ++i) {
    const Node* p_cur_node = graph_utils::GetInputNode(concat, i);
    if (p_cur_node != nullptr) {
      graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *p_cur_node);
    }
  }

  if (!graph_utils::ReplaceNodeWithInitializer(graph, *graph.GetNode(concat.Index()), new_node_arg)) {
    return false;
  }
  return true;
}

}  // namespace onnxruntime
