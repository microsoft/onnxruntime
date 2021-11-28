// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/additive_mask_softmax_dropout.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status AdditiveMaskSoftmaxDropoutFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    std::vector<std::reference_wrapper<Node>> nodes_to_fuse;

    // matching for softmax node
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Softmax", {1, 11}, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
        // std::cout << "111111111111111111" << std::endl;
      continue;
    }
  
    // std::cout << "2222222222222222222" << std::endl;
    std::vector<NodeArg*> fused_op_inputs, fused_op_ouputs;
    // const TensorShapeProto* input1_shape = node.MutableInputDefs()[0]->Shape();
    // const TensorShapeProto* input2_shape = node.MutableInputDefs()[1]->Shape();
    // const TensorShapeProto* input3_shape = node.MutableInputDefs()[2]->Shape();

    // if (input1_shape == nullptr ||
    //     input2_shape == nullptr ||
    //     input3_shape == nullptr ||
    //     input1_shape->dim_size() < 1 ||
    //     input2_shape->dim_size() != 0 ||
    //     input3_shape->dim_size() < 1) {
    //   continue;
    // }

    // check broadcastable...
    // if (!IsSameShape(*input1_shape, *input3_shape)) {
    //   continue;
    // }


    // if (input2_shape->dim_size() != 0) {
    //   continue;
    // }

    // int last_dim_shape1 = input1_shape->dim_size() - 1;
    // int last_dim_shape2 = input2_shape->dim_size() - 1;
    // if (!utils::HasDimValue(input1_shape->dim(last_dim_shape1)) ||
    //     !utils::HasDimValue(input2_shape->dim(last_dim_shape2)) ||
    //     input1_shape->dim(last_dim_shape1).dim_value() != input2_shape->dim(last_dim_shape2).dim_value()) {
    //   continue;
    // }


    auto input_arg = node.MutableInputDefs()[0];
    fused_op_inputs.push_back(input_arg);

    auto& mask_input = graph.GetOrCreateNodeArg(input_arg->Name() + "_mask", input_arg->TypeAsProto());
    fused_op_inputs.push_back(&mask_input);

    auto softmax_ret_type_proto = *node.MutableOutputDefs()[0]->TypeAsProto();
    auto& softmax_result_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("softmax_result"), &softmax_ret_type_proto);
    fused_op_ouputs.push_back(&softmax_result_arg);

    nodes_to_fuse.push_back(node);

    // matching for Dropout node
    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
        // std::cout << "333333333333333333333333" << std::endl;
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Dropout", {12}, kOnnxDomain)) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
            // std::cout << "4444444444444444444" << std::endl;
      continue;
    }

    // todo: check training mode = True

    if (graph.NodeProducesGraphOutput(node)) {
        // std::cout << "555555555555555555555" << std::endl;
      continue;
    }

    std::cout << "66666666666666666" << std::endl;
    Node& dropout_node = *graph.GetNode(next_node.Index());
    nodes_to_fuse.push_back(dropout_node);

    fused_op_inputs.push_back(dropout_node.MutableInputDefs()[1]);

    auto dropout_ret_type_proto = *dropout_node.MutableOutputDefs()[0]->TypeAsProto();
    auto& dropout_result_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("dropout_result"), &dropout_ret_type_proto);
    auto dropout_mask_type_proto = *dropout_node.MutableOutputDefs()[1]->TypeAsProto();
    auto& dropout_mask_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("dropout_mask"), &dropout_mask_type_proto);

    fused_op_ouputs.push_back(&dropout_result_arg);
    fused_op_ouputs.push_back(&dropout_mask_arg);

    const std::string op_type = "AdditiveMaskSoftmaxDropout";
    Node& fused_op_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                                  op_type,
                                                  "fused Softmax+Dropout for " + node.Name(),
                                                  fused_op_inputs,
                                                  fused_op_ouputs,
                                                  nullptr,
                                                  kMSDomain);


    // Get attribute "seed" from "Softmax" node if available.

    NodeAttributes::const_iterator seed = dropout_node.GetAttributes().find("seed");
    if (seed != dropout_node.GetAttributes().end()) {
      fused_op_node.AddAttribute("seed", seed->second);
    }

    // NodeAttributes::const_iterator seed = dropout_node.GetAttributes().find("seed");
    // if (seed != dropout_node.GetAttributes().end()) {
    //   fused_op_node.AddAttribute("seed", seed->second);
    // }


    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_op_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::ReplaceDownstreamNodeInput(graph, node, 0, fused_op_node, 0);
    graph_utils::ReplaceDownstreamNodeInput(graph, dropout_node, 0, fused_op_node, 1);
    graph_utils::ReplaceDownstreamNodeInput(graph, dropout_node, 1, fused_op_node, 2);
    // const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, 0);
    // if (nullptr == edge) {  // handle input/initializer
    //     graph_utils::ReplaceNodeInput(node, 0, *(mlp_f_node.MutableOutputDefs()[0]));
    // } else {
    //     auto input_node = const_cast<Node*>(&edge->GetNode());
    //     graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), mlp_f_node, 0);
    // }

    // graph_utils::FinalizeNodeFusion(graph, {node, dropout_node}, fused_op_node);

    // // delete bias_add_node, softmax_node and optionally residual_add_node
    for (Node& n : nodes_to_fuse) {
      graph_utils::RemoveNodeOutputEdges(graph, n);
      graph.RemoveNode(n.Index());
    }

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime