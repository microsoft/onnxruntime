// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/sce_loss_grad_bias_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

/**
Fuse SoftmaxCrossEntropyLossInternalGrad + Reshape(optional) + Sum/Add to SoftmaxCrossEntropyLossInternalGrad.
If it's Sum Op, it requires that it has only 2 inputs. Sum/Add must be non-broadcasting computation.
*/
Status SceLossGradBiasFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLossInternalGrad", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) || node.InputDefs().size() == 6 ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    NodeArg* sce_grad_def = node.MutableOutputDefs()[0];
    Node* p_next = graph.GetNode(node.OutputNodesBegin()->Index());
    Node* p_reshape = nullptr;
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*p_next, "Reshape", {5, 13, 14}) &&
        graph_utils::IsSupportedProvider(*p_next, GetCompatibleExecutionProviders()) &&
        p_next->GetOutputEdgesCount() == 1) {
      p_reshape = p_next;
      sce_grad_def = p_reshape->MutableOutputDefs()[0];
      p_next = graph.GetNode(p_next->OutputNodesBegin()->Index());
    }

    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(*p_next, "Add", {7, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(*p_next, "Sum", {6, 8, 13})) ||
        !graph_utils::IsSupportedProvider(*p_next, GetCompatibleExecutionProviders()) ||
        p_next->InputDefs().size() != 2) {
      continue;
    }

    Node& sum_node = *p_next;
    auto& sum_input_defs = sum_node.MutableInputDefs();
    auto shape0 = sum_input_defs[0]->Shape();
    auto shape1 = sum_input_defs[1]->Shape();
    if (!shape0 || !shape1 || shape0->dim_size() != shape1->dim_size()) {
      continue;
    }

    bool has_same_shape = true;
    for (int i = 0; i < shape0->dim_size(); ++i) {
      if (shape0->dim(i) != shape1->dim(i)) {
        has_same_shape = false;
        break;
      }
    }

    if (!has_same_shape) continue;

    NodeArg* bias_def = sce_grad_def == sum_input_defs[0] ? sum_input_defs[1] : sum_input_defs[0];
    auto& scegrad_inputs = node.MutableInputDefs();
    InlinedVector<NodeArg*> new_scegrad_node_inputs{scegrad_inputs[0], scegrad_inputs[1], scegrad_inputs[2]};
    InlinedVector<NodeArg*> new_scegrad_node_outputs;
    if (scegrad_inputs.size() >= 4) {
      new_scegrad_node_inputs.emplace_back(scegrad_inputs[3]);
    } else {
      new_scegrad_node_inputs.emplace_back(&graph.GetOrCreateNodeArg("", nullptr));
    }
    if (scegrad_inputs.size() >= 5) {
      new_scegrad_node_inputs.emplace_back(scegrad_inputs[4]);
    } else {
      ONNX_NAMESPACE::TensorProto ignore_index_initializer_proto;
      ignore_index_initializer_proto.set_name(graph.GenerateNodeArgName("sce_grad_ignore_index"));
      ignore_index_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      ignore_index_initializer_proto.add_int64_data(static_cast<int64_t>(-1));
      new_scegrad_node_inputs.emplace_back(&graph_utils::AddInitializer(graph, ignore_index_initializer_proto));
    }
    new_scegrad_node_inputs.emplace_back(bias_def);
    if (!p_reshape) {
      new_scegrad_node_outputs.emplace_back(sum_node.MutableOutputDefs()[0]);
    } else {
      new_scegrad_node_outputs.emplace_back(p_reshape->MutableInputDefs()[0]);
    }
    Node& new_scegrad_node =
        graph.AddNode(graph.GenerateNodeName("FusedSoftmaxCrossEntropyLossInternalGrad"),
                      "SoftmaxCrossEntropyLossInternalGrad", "FusedSoftmaxCrossEntropyLossInternalGrad",
                      new_scegrad_node_inputs, new_scegrad_node_outputs, &node.GetAttributes(), kMSDomain);
    new_scegrad_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
    if (p_reshape) {
      graph_utils::FinalizeNodeFusion(graph, *p_reshape, sum_node);
    } else {
      graph_utils::RemoveNodeOutputEdges(graph, sum_node);
      graph.RemoveNode(sum_node.Index());
    }

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
