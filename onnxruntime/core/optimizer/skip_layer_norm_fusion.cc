// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

/**
Skip Layer Normalization will fuse Add + LayerNormalization into one node.
*/
Status SkipLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_add = graph.GetNode(node_index);
    if (p_add == nullptr)
      continue;  // we removed the node as part of an earlier fusion.

    Node& add_node = *p_add;
    ORT_RETURN_IF_ERROR(Recurse(add_node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
        !graph_utils::IsSupportedProvider(add_node, GetCompatibleExecutionProviders()) ||
        add_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(add_node)) {
      continue;
    }

    // Check the input dimensions of the "Add" node.
    const TensorShapeProto* add_input1_shape = add_node.MutableInputDefs()[0]->Shape();
    const TensorShapeProto* add_input2_shape = add_node.MutableInputDefs()[1]->Shape();

    if (add_input1_shape == nullptr || add_input2_shape == nullptr) {
      continue;
    }
    // "Add" inputs have to be 3d.
    if (add_input1_shape->dim_size() != 3 || add_input2_shape->dim_size() != 3) {
      continue;
    }
    // "Add" inputs have to be of same dimensions.
    bool isValidInput = true;
    for (int i = 0; i < 3; i++) {
      if (add_input1_shape->dim(i).dim_value() != add_input2_shape->dim(i).dim_value()) {
        isValidInput = false;
        break;
      }
    }
    if (!isValidInput) {
      continue;
    }

    // Find "LayerNormalization" node after the "Add".
    Node& ln_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(ln_node, "LayerNormalization", {1}) ||
        ln_node.GetExecutionProviderType() != add_node.GetExecutionProviderType() ||
        !IsSupportedDataType(ln_node)) {
      continue;
    }

    // Find Previous add node
    //NodeArg* p_bias_node_arg = nullptr;
    //NodeArg* p_input_node_arg = nullptr;
    Node* p_bias_node = nullptr;
    for (auto node_iter = add_node.InputNodesBegin(); node_iter != add_node.InputNodesEnd(); ++node_iter) {
      Node* p_node = graph.GetNode(node_iter->Index());
      if (!p_node ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*p_node, "Add", {7}) ||
          !graph_utils::IsSupportedProvider(*p_node, GetCompatibleExecutionProviders()) ||
          p_node->GetOutputEdgesCount() != 1 ||
          !IsSupportedDataType(*p_node)) {
        continue;
      }

      p_bias_node = p_node;
      break;
      /*const TensorShapeProto* bias_node_input1_shape = p_bias_node->MutableInputDefs()[0]->Shape();
      const TensorShapeProto* bias_node_input2_shape = p_bias_node->MutableInputDefs()[1]->Shape();
      if (bias_node_input1_shape != nullptr &&
          bias_node_input1_shape->dim_size() == 1 &&
          graph_utils::NodeArgIsConstant(graph, *(p_bias_node->InputDefs()[0]))) {
        p_bias_node_arg = p_bias_node->MutableInputDefs()[0];
        p_input_node_arg = p_bias_node->MutableInputDefs()[1];
        break;
      }
      if (bias_node_input2_shape != nullptr &&
          bias_node_input2_shape->dim_size() == 1 &&
          graph_utils::NodeArgIsConstant(graph, *(p_bias_node->InputDefs()[1]))) {
        p_input_node_arg = p_bias_node.MutableInputDefs()[0];
        p_bias_node_arg = p_bias_node.MutableInputDefs()[1];
        break;
      }*/
    }

    // Get the inputs for the new SkipLayerNormalization node.
    std::vector<NodeArg*> skip_layer_norm_input_defs{add_node.MutableInputDefs()[0],
                                                     add_node.MutableInputDefs()[1],
                                                     ln_node.MutableInputDefs()[1],
                                                     ln_node.MutableInputDefs()[2]};

    if (p_bias_node != nullptr) {
      skip_layer_norm_input_defs[0] = p_bias_node->MutableInputDefs()[0];
      skip_layer_norm_input_defs.push_back(p_bias_node->MutableInputDefs()[1]);
      nodes_to_remove.push_back(*p_bias_node);
    }

    nodes_to_remove.push_back( add_node );
    nodes_to_remove.push_back( ln_node );

    Node& skip_layer_norm_node = graph.AddNode(graph.GenerateNodeName("SkipLayerNormalization"),
                                               "SkipLayerNormalization",
                                               "fused SkipLayerNorm subgraphs ",
                                               skip_layer_norm_input_defs,
                                               {}, {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    skip_layer_norm_node.SetExecutionProviderType(add_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the layer_norm_node.
    // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, skip_layer_norm_node);

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime
