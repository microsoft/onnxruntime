// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// Get values of an constant integer tensor from input, and append them to a vector.
static bool GetConstantInput(const Graph& graph, const NodeArg& input_arg, std::vector<int64_t>& data) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  if (tensor_proto == nullptr) {
    return false;
  }

  auto init_const = onnxruntime::make_unique<Initializer>(*tensor_proto);
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    const int64_t* val = init_const->data<int64_t>();
    data.reserve(data.size() + init_const->size());
    data.insert(data.end(), val, val + init_const->size());
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    const int32_t* val = init_const->data<int32_t>();
    data.reserve(data.size() + init_const->size());
    for (int64_t i = 0; i < init_const->size(); i++) {
      data.push_back(static_cast<int64_t>(val[i]));
    }
  } else {
    return false;
  }

  return true;
}

Status ReshapeFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_reshape = graph.GetNode(node_index);
    if (p_reshape == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reshape = *p_reshape;
    ORT_RETURN_IF_ERROR(Recurse(reshape, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reshape, "Reshape", {5}) ||
        !graph_utils::IsSupportedProvider(reshape, GetCompatibleExecutionProviders())) {
      continue;
    }

    const Node* p_root = graph_utils::GetInputNode(reshape, 0);

    const Node* p_concat = graph_utils::GetInputNode(reshape, 1);
    if (nullptr == p_concat) {
      continue;
    }
    const Node& concat = *p_concat;

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(concat, "Concat", {1, 4})) {
      continue;
    }

    auto concat_input_count = concat.InputArgCount().front();
    if (concat_input_count < 3 || concat_input_count > 4 || concat.GetOutputEdgesCount() > 1) {
      continue;
    }

    // path 1: [Root] --> Shape --> Gather(indices=0) --> Unsqueeze (axes=0) --> Concat [input 0]
    std::vector<graph_utils::MatchEdgeInfo> parent_path {
        {0, "Unsqueeze", {1}, kOnnxDomain},
        {0, "Gather", {1}, kOnnxDomain},
        {0, "Shape", {1}, kOnnxDomain}
    };

    std::vector<const Node::EdgeEnd*> edges;
    if (!graph_utils::FindParentPath(concat, parent_path, edges)) {
      continue;
    }
    
    const Node& unsqueeze_1 = edges[0]->GetNode();
    const Node& gather_1 = edges[1]->GetNode();
    const Node& shape_1 = edges[2]->GetNode();
    if (unsqueeze_1.GetOutputEdgesCount() != 1 || gather_1.GetOutputEdgesCount() != 1 || shape_1.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (graph_utils::GetInputNode(shape_1, 0) != p_root) {
      continue;
    }

    std::vector<int64_t> axes;
    if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze_1, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
      continue;
    }

    if (!optimizer_utils::CheckConstantInput(graph, *(gather_1.InputDefs()[1]), int(0))) {
      continue;
    }

    // path 2: [Root] --> Shape --> Gather(indices=1) --> Unsqueeze (axes=0) --> Concat [input 1]
    parent_path[0].dst_arg_index = 1;

    if (!graph_utils::FindParentPath(concat, parent_path, edges)) {
      continue;
    }

    const Node& unsqueeze_2 = edges[0]->GetNode();
    const Node& gather_2 = edges[1]->GetNode();
    const Node& shape_2 = edges[2]->GetNode();
    if (unsqueeze_2.GetOutputEdgesCount() != 1 || gather_2.GetOutputEdgesCount() != 1 || shape_2.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (graph_utils::GetInputNode(shape_2, 0) != p_root) {
      continue;
    }

    if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze_2, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
      continue;
    }

    if (!optimizer_utils::CheckConstantInput(graph, *(gather_2.InputDefs()[1]), int(1))) {
      continue;
    }

    // Compose the shape value input for reshape op.
    std::vector<int64_t> shape_value = {0, 0};

    if (!graph_utils::NodeArgIsConstant(graph, *(concat.InputDefs()[2])) ||
        !GetConstantInput(graph, *(concat.InputDefs()[2]), shape_value)) {
      continue;
    }

    if (concat_input_count > 3) {
      if (!graph_utils::NodeArgIsConstant(graph, *(concat.InputDefs()[3])) ||
          !GetConstantInput(graph, *(concat.InputDefs()[3]), shape_value)) {
        continue;
      }
    }

    // Create an initializer with the same name as the concat node output, and replace the concat node
    const auto& new_initializer_name = concat.OutputDefs()[0]->Name();
    if (!graph_utils::CanReplaceNodeWithInitializer(graph, concat, new_initializer_name)) {
      continue;
    }
    const auto* shape_def = concat.OutputDefs()[0];
    ONNX_NAMESPACE::TensorProto shape_initializer_proto;
    shape_initializer_proto.set_name(shape_def->Name());
    shape_initializer_proto.add_dims(static_cast<int64_t>(shape_value.size()));
    shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_initializer_proto.set_raw_data(shape_value.data(), shape_value.size() * sizeof(int64_t));
    auto& new_node_arg = graph_utils::AddInitializer(graph, shape_initializer_proto);
    if (!graph_utils::ReplaceNodeWithInitializer(graph, *graph.GetNode(concat.Index()), new_node_arg)) {
      continue;
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

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
