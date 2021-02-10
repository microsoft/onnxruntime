// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(bfloat16)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

static bool CheckFirstAdd(Node& add, ProviderType providertype) {
  if (providertype != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) ||
      add.GetOutputEdgesCount() != 1) {
    return false;
  }

  // Check the input dimensions of the "Add" node.
  const TensorShapeProto* add_input1_shape = add.MutableInputDefs()[0]->Shape();
  const TensorShapeProto* add_input2_shape = add.MutableInputDefs()[1]->Shape();

  if (add_input1_shape == nullptr || add_input2_shape == nullptr) {
    return false;
  }
  // "Add" inputs have to be 3d.
  if (add_input1_shape->dim_size() != 3 || add_input2_shape->dim_size() != 3) {
    return false;
  }
  // "Add" inputs have to be of same dimensions.
  bool is_valid_input = true;
  for (int i = 0; i < 3; i++) {
    if (!utils::HasDimValue(add_input1_shape->dim(i)) ||
        !utils::HasDimValue(add_input2_shape->dim(i)) ||
        add_input1_shape->dim(i).dim_value() != add_input2_shape->dim(i).dim_value()) {
      is_valid_input = false;
      break;
    }
  }
  return is_valid_input;
}

// Add2 is the 2nd add of the to be fused sub-graph
// The 1st input should be a 3D tensor
// The 2nd input should be a 1D constant value
static bool CheckSecondAdd(Graph& graph, Node& add, ProviderType providertype) {
  if (providertype != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) ||
      add.GetOutputEdgesCount() != 1) {
    return false;
  }

  // The 2nd input should be a constant value
  if (!graph_utils::NodeArgIsConstant(graph, *(add.MutableInputDefs()[1]))) {
    return false;
  }

  // Check the input dimensions of the "Add" node.
  const TensorShapeProto* add_input1_shape = add.MutableInputDefs()[0]->Shape();
  const TensorShapeProto* add_input2_shape = add.MutableInputDefs()[1]->Shape();

  if (add_input1_shape == nullptr || add_input2_shape == nullptr) {
    return false;
  }

  return add_input1_shape->dim_size() == 3 &&
         add_input2_shape->dim_size() == 1 &&
         utils::HasDimValue(add_input1_shape->dim(2)) &&
         utils::HasDimValue(add_input2_shape->dim(0)) &&
         add_input1_shape->dim(2).dim_value() == add_input2_shape->dim(0).dim_value();
}

/**
Skip Layer Normalization will fuse Add + LayerNormalization into one node, and another Add if applicable

Before fusion:
Format 1:
    [Sub1]  C    [Sub2]
        \  /     /
        Add2    /
           \   /
            Add1
             |
     LayerNormalization

Format 2:
      [Sub1] [Sub2]  C
         \      \   /
          \     Add2
           \    /
            Add1
             |
     LayerNormalization

Format 3:
      [Sub1]   [Sub2]
         \       /
          \     /
           \   /
            Add1
             |
     LayerNormalization

After fusion:
       [Sub1]   [Sub1]
         \      /
          \    /
    SkipLayerNormalization

Note: This fusion doesn't consider the following case:
      [Sub1]   [Sub2]
         \       /
        Add2  Add3
           \   /
            Add1
             |
     LayerNormalization
*/

Status SkipLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    Node* p_layernorm = graph.GetNode(node_index);
    if (p_layernorm == nullptr)
      continue;  // node was removed in an earlier fusion.

    Node& ln_node = *p_layernorm;
    ORT_RETURN_IF_ERROR(Recurse(ln_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(ln_node, "LayerNormalization", {1}) ||
        !graph_utils::IsSupportedProvider(ln_node, GetCompatibleExecutionProviders()) ||
        !IsSupportedDataType(ln_node)) {
      continue;
    }

    enum class Format : int8_t {
      Format1,
      Format2,
      Format3,
      None
    };

    Node* p_add1 = nullptr;
    Node* p_add2 = nullptr;
    Format matched_format = Format::None;

    // Format 1
    std::vector<graph_utils::EdgeEndToMatch> format1_parent_path{
        {0, 0, "Add", {7, 13}, kOnnxDomain},
        {0, 0, "Add", {7, 13}, kOnnxDomain}};

    std::vector<const Node::EdgeEnd*> edges;
    if (graph_utils::FindPath(ln_node, true, format1_parent_path, edges, logger)) {
      p_add1 = const_cast<Node*>(&edges[0]->GetNode());
      p_add2 = const_cast<Node*>(&edges[1]->GetNode());

      if (CheckFirstAdd(*p_add1, ln_node.GetExecutionProviderType()) &&
          CheckSecondAdd(graph, *p_add2, ln_node.GetExecutionProviderType()) &&
          graph.GetNodeOutputsInGraphOutputs(*p_add1).empty() &&
          graph.GetNodeOutputsInGraphOutputs(*p_add2).empty()) {
        matched_format = Format::Format1;
      }
    }

    if (matched_format == Format::None) {
      // Format 2
      std::vector<graph_utils::EdgeEndToMatch> format2_parent_path{
          {0, 0, "Add", {7, 13}, kOnnxDomain},
          {0, 1, "Add", {7, 13}, kOnnxDomain}};

      if (graph_utils::FindPath(ln_node, true, format2_parent_path, edges, logger)) {
        p_add1 = const_cast<Node*>(&edges[0]->GetNode());
        p_add2 = const_cast<Node*>(&edges[1]->GetNode());

        if (CheckFirstAdd(*p_add1, ln_node.GetExecutionProviderType()) &&
            CheckSecondAdd(graph, *p_add2, ln_node.GetExecutionProviderType()) &&
            graph.GetNodeOutputsInGraphOutputs(*p_add1).empty() &&
            graph.GetNodeOutputsInGraphOutputs(*p_add2).empty()) {
          matched_format = Format::Format2;
        }
      }
    }

    if (matched_format == Format::None) {
      // Format 3
      std::vector<graph_utils::EdgeEndToMatch> format3_parent_path{
          {0, 0, "Add", {7, 13}, kOnnxDomain}};

      if (graph_utils::FindPath(ln_node, true, format3_parent_path, edges, logger)) {
        p_add1 = const_cast<Node*>(&edges[0]->GetNode());

        if (CheckFirstAdd(*p_add1, ln_node.GetExecutionProviderType()) &&
            graph.GetNodeOutputsInGraphOutputs(*p_add1).empty()) {
          matched_format = Format::Format3;
        }
      }
    }

    if (matched_format == Format::None) {
      continue;
    }

    NodeArg beta_place_holder("", nullptr);

    // Get the inputs for the new SkipLayerNormalization node.
    std::vector<NodeArg*> skip_layer_norm_input_defs{p_add1->MutableInputDefs()[0],
                                                     p_add1->MutableInputDefs()[1],
                                                     ln_node.MutableInputDefs()[1],
                                                     ln_node.MutableInputDefs().size() == 2 ? &beta_place_holder : ln_node.MutableInputDefs()[2]};

    if (matched_format == Format::Format1) {
      skip_layer_norm_input_defs[0] = p_add2->MutableInputDefs()[0];
      skip_layer_norm_input_defs.push_back(p_add2->MutableInputDefs()[1]);
      nodes_to_remove.push_back(*p_add2);
    } else if (matched_format == Format::Format2) {
      skip_layer_norm_input_defs[1] = p_add2->MutableInputDefs()[0];
      skip_layer_norm_input_defs.push_back(p_add2->MutableInputDefs()[1]);
      nodes_to_remove.push_back(*p_add2);
    }

    nodes_to_remove.push_back(*p_add1);
    nodes_to_remove.push_back(ln_node);

    Node& skip_layer_norm_node = graph.AddNode(graph.GenerateNodeName("SkipLayerNormalization"),
                                               "SkipLayerNormalization",
                                               "fused SkipLayerNorm subgraphs ",
                                               skip_layer_norm_input_defs,
                                               ln_node.MutableOutputDefs(), {}, kMSDomain);
    // Get attribute "epsilon" from "LayerNormalization" node if available. Else, default value
    // will be used.
    NodeAttributes ln_attrs = ln_node.GetAttributes();
    NodeAttributes::const_iterator epsilon = ln_attrs.find("epsilon");
    if (epsilon != ln_attrs.end()) {
      skip_layer_norm_node.AddAttribute("epsilon", epsilon->second);
    } else {
      skip_layer_norm_node.AddAttribute("epsilon", contrib::kDefaultSkipLayerNormEpsilon);
    }
    // Assign provider to this new node. Provider should be same as the provider for old node.
    skip_layer_norm_node.SetExecutionProviderType(ln_node.GetExecutionProviderType());
  }
  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  modified = true;

  return Status::OK();
}
}  // namespace onnxruntime
