// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/bias_softmax_dropout_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

// Both BiasSoftmax and BiasSoftmaxDropout follow the axis attribute definition from Softmax Op before OpSet-11,
// that it, no extra transpose if axis is not rank-1. So if BiasSoftmax is found in the graph, it's safe to
// fuse matched sub-graph to BiasSoftmaxDropout without checking the axis attribute.
Status BiasSoftmaxDropoutFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                           const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Matching for BiasSoftmax node.
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "BiasSoftmax", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    Node* p_dropout = nullptr;
    Node* p_softmax_grad = nullptr;
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      Node& next_node = *graph.GetNode(it->Index());
      if ((graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Dropout", {12, 13}) ||
           graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "BitmaskDropout", {1}, kMSDomain)) &&
          graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders())) {
        p_dropout = &next_node;
      } else if ((graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "SoftmaxGrad", {1}, kMSDomain) ||
                  graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "SoftmaxGrad_13", {1}, kMSDomain)) &&
                 graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders())) {
        p_softmax_grad = &next_node;
      }
    }

    // BiasSoftmaxDropout is for training only so that the Dropout node must have ratio and training_mode=True inputs,
    // and must have mask/bitmask output.
    if (!p_dropout || !p_softmax_grad || p_dropout->InputDefs().size() < 3 || p_dropout->OutputDefs().size() < 2) {
      continue;
    }
    const ONNX_NAMESPACE::TensorProto* initializer =
        graph_utils::GetConstantInitializer(graph, p_dropout->InputDefs()[2]->Name());
    if (!initializer || !(*Initializer(*initializer, graph.ModelPath()).data<bool>())) continue;

    Node& dropout_node = *p_dropout;
    Node& softmax_grad_node = *p_softmax_grad;

    Node* p_dropout_grad = nullptr;
    for (auto it = dropout_node.OutputNodesBegin(); it != dropout_node.OutputNodesEnd(); ++it) {
      Node& next_node = *graph.GetNode(it->Index());
      if ((graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "DropoutGrad", {1}, kMSDomain) ||
           graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "BitmaskDropoutGrad", {1}, kMSDomain)) &&
          graph_utils::IsSupportedProvider(next_node, GetCompatibleExecutionProviders())) {
        p_dropout_grad = &next_node;
        break;
      }
    }

    if (!p_dropout_grad) continue;
    Node& dropout_grad_node = *p_dropout_grad;

    InlinedVector<NodeArg*> bias_softmax_dropout_inputs{node.MutableInputDefs()[0], node.MutableInputDefs()[1],
                                                        dropout_node.MutableInputDefs()[1]};         // [input, bias, ratio]
    InlinedVector<NodeArg*> bias_softmax_dropout_outputs;                                            // [y, dropout_mask, softmax_y]
    InlinedVector<NodeArg*> softmax_dropout_grad_inputs;                                             // [dy, dropout_mask, softmax_y, ratio]
    InlinedVector<NodeArg*> softmax_dropout_grad_outputs{softmax_grad_node.MutableOutputDefs()[0]};  // [dx]
    bias_softmax_dropout_outputs.emplace_back(dropout_node.MutableOutputDefs()[0]);                  // y
    softmax_dropout_grad_inputs.emplace_back(dropout_grad_node.MutableInputDefs()[0]);               // dy
    if (dropout_node.OpType() == "BitmaskDropout") {
      // BiasSoftmaxDropout supports bool mask only. If it's BitmaskDropout, need to create a new bool NodeArg.
      ONNX_NAMESPACE::TypeProto tensor_bool;
      tensor_bool.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
      NodeArg& mask_output_def =
          graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("bias_softmax_dropout_mask"), &tensor_bool);
      bias_softmax_dropout_outputs.emplace_back(&mask_output_def);  // dropout_mask
      softmax_dropout_grad_inputs.emplace_back(&mask_output_def);   // dropout_mask
    } else {
      bias_softmax_dropout_outputs.emplace_back(dropout_node.MutableOutputDefs()[1]);     // dropout_mask
      softmax_dropout_grad_inputs.emplace_back(dropout_grad_node.MutableInputDefs()[1]);  // dropout_mask
    }
    bias_softmax_dropout_outputs.emplace_back(node.MutableOutputDefs()[0]);             // softmax_y
    softmax_dropout_grad_inputs.emplace_back(softmax_grad_node.MutableInputDefs()[1]);  // softmax_y
    softmax_dropout_grad_inputs.emplace_back(dropout_grad_node.MutableInputDefs()[2]);  // ratio
    NodeAttributes bias_softmax_dropout_attrs;                                          // {axis, is_inner_broadcast, seed}
    for (const auto& pair : node.GetAttributes()) {
      bias_softmax_dropout_attrs[pair.first] = pair.second;
    }
    for (const auto& pair : dropout_node.GetAttributes()) {
      bias_softmax_dropout_attrs[pair.first] = pair.second;
    }
    Node& bias_softmax_dropout_node = graph.AddNode(
        graph.GenerateNodeName("BiasSoftmaxDropout"), "BiasSoftmaxDropout", "BiasSoftmaxDropout",
        bias_softmax_dropout_inputs, bias_softmax_dropout_outputs, &bias_softmax_dropout_attrs, kMSDomain);
    Node& softmax_droput_grad_node = graph.AddNode(
        graph.GenerateNodeName("SoftmaxDropoutGrad"), "SoftmaxDropoutGrad", "SoftmaxDropoutGrad",
        softmax_dropout_grad_inputs, softmax_dropout_grad_outputs, &softmax_grad_node.GetAttributes(), kMSDomain);
    bias_softmax_dropout_node.SetExecutionProviderType(node.GetExecutionProviderType());
    softmax_droput_grad_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, dropout_node);
    graph.RemoveNode(dropout_node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, dropout_grad_node);
    graph.RemoveNode(dropout_grad_node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, softmax_grad_node);
    graph.RemoveNode(softmax_grad_node.Index());
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
