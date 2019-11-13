// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(double)"};

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
Layer Normalization will fuse LayerNormalization into one node : 
+---------------------+ 
|                     |
|                     v 
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add 
                      |                                               ^ 
                      |                                               |
                      +-----------------------------------------------+
It also handles cases of duplicated sub nodes exported from older version of PyTorch : 
+---------------------+ 
|                     v 
|          +-------> Sub ---------------------------------------------+ 
|          |                                                          |
|          |                                                          v 
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add 
|                     ^
|                     |
+---------------------+
*/
Status LayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_reduce_mean = graph.GetNode(node_index);
    if (p_reduce_mean == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reduce_mean_node = *p_reduce_mean;
    ORT_RETURN_IF_ERROR(Recurse(reduce_mean_node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_node, "ReduceMean", {1}) ||
        !graph_utils::IsSupportedProvider(reduce_mean_node, GetCompatibleExecutionProviders()) ||
        (reduce_mean_node.GetOutputEdgesCount() != 1 && reduce_mean_node.GetOutputEdgesCount() != 2) ||
        !IsSupportedDataType(reduce_mean_node)) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean_node);

    // Loop through the children of current "ReduceMean" node. See if they match ["Sub"] or ["Sub", "Sub"]
    int subCnt = 0;
    const Node* p_sub_node = nullptr;
    const Node* p_sub_node_dup = nullptr;
    for (auto iter = reduce_mean_node.OutputNodesBegin(); iter != reduce_mean_node.OutputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Sub") == 0) {
        if (subCnt == 0) {
          p_sub_node = &(*iter);

        } else {
          p_sub_node_dup = &(*iter);
        }
        subCnt++;
      } else {
        // doesn't match layer norm pattern. break.
        subCnt = -1;
        break;
      }
    }

    if (subCnt != 1 && subCnt != 2) {
      continue;
    }

    Node& sub_node = *graph.GetNode(p_sub_node->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node, "Sub", {7}) ||
        sub_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(sub_node)) {
      continue;
    }
    nodes_to_remove.push_back(sub_node);
    
    // Find the "Div" node after "Sub".
    const Node* p_div = nullptr;
    p_div = graph_utils::FirstChildByType(sub_node, "Div");

    // Find the sub_dup node if exist
    if (p_sub_node_dup != nullptr) {
      Node& sub_node_dup = *graph.GetNode(p_sub_node_dup->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node_dup, "Sub", {7}) ||
          sub_node_dup.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
          sub_node_dup.GetOutputEdgesCount() != 1 ||
          !IsSupportedDataType(sub_node_dup)) {
        continue;
      }
      nodes_to_remove.push_back(sub_node_dup);
      // Find Div node after the duplicated sub node if it's not found after the first sub node. 
      if (p_div == nullptr) {
        p_div = graph_utils::FirstChildByType(sub_node_dup, "Div");
      }
    }

    if (p_div == nullptr) {
      continue;
    }
    Node& div_node = *graph.GetNode(p_div->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7}) ||
        div_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        div_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(div_node)) {
      continue;
    }
    nodes_to_remove.push_back(div_node);

    // Traceback the div node to find sqrt --> div
    const Node* p_sqrt = graph_utils::FirstParentByType(div_node, "Sqrt");
    if (p_sqrt == nullptr) {
      continue;
    }
    Node& sqrt_node = *graph.GetNode(p_sqrt->Index());

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6}) ||
        sqrt_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        sqrt_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(sqrt_node) || 
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(sqrt_node);

    // Traceback the sqrt node to find add --> sqrt
    Node& add2_node = *graph.GetNode(sqrt_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7}) ||
        add2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        add2_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(add2_node)) {
      continue;
    }
    nodes_to_remove.push_back(add2_node);

    // Traceback the add node to find reduceMean --> add
    const Node* p_reduce_mean2 = nullptr;

    p_reduce_mean2 = graph_utils::FirstParentByType(add2_node, "ReduceMean");
    if (p_reduce_mean2 == nullptr) {
      continue;
    }
    Node& reduce_mean2_node = *graph.GetNode(p_reduce_mean2->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean2_node, "ReduceMean", {1}) ||
        reduce_mean2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        reduce_mean2_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(reduce_mean2_node) || 
        reduce_mean2_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean2_node);

    // Traceback the reduceMean node to find pow --> reduceMean
    Node& pow_node = *graph.GetNode(reduce_mean2_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7}) ||
        pow_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        pow_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(pow_node)) {
      continue;
    }
    nodes_to_remove.push_back(pow_node);

    // Traceback the pow node to find sub --> pow
    const Node* p_sub2_node = graph_utils::FirstParentByType(pow_node, "Sub");
    if (p_sub2_node == nullptr) {
      continue;
    }
    Node& sub2_node = *graph.GetNode(p_sub2_node->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub2_node, "Sub", {7}) ||
        sub2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(sub2_node)) {
      continue;
    }

    // Traceback the sub node to find reduceMean --> sub
    const Node* p_reduce_mean_check = graph_utils::FirstParentByType(sub2_node, "ReduceMean");
    // Check if the reduceMean node after traceback is the same node as the reduceMean node from the beginning.
    if (p_reduce_mean_check == nullptr || p_reduce_mean_check != &reduce_mean_node) {
      continue;
    }

    // div --> mul
    Node& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        mul_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }
    nodes_to_remove.push_back(mul_node);

    // mul --> add
    Node& last_add_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(last_add_node, "Add", {7}) ||
        last_add_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(last_add_node)) {
      continue;
    }
    nodes_to_remove.push_back(last_add_node);

    // Get the inputs for the new LayerNormalization node.
    NodeArg* scale = nullptr;
    NodeArg* bias = nullptr;
    for (size_t i = 0; i < mul_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(mul_node.MutableInputDefs()[i])) || 
          graph_utils::IsGraphInput(graph, mul_node.MutableInputDefs()[i])) {
        // Scale must be 1d.
        if (mul_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          scale = mul_node.MutableInputDefs()[i];
        }
      }
    }

    for (size_t i = 0; i < last_add_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(last_add_node.MutableInputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, last_add_node.MutableInputDefs()[i])) {
        // Bias must be 1d.
        if (last_add_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          bias = last_add_node.MutableInputDefs()[i];
        }
      }
    }
    if (scale == nullptr || bias == nullptr) {
      continue;
    }

    // Scale and bias must have the same dimension. 
    if (scale->Shape()->dim(0).dim_value() != bias->Shape()->dim(0).dim_value()) {
      continue;
    }
    const std::vector<NodeArg*> layer_norm_input_defs{reduce_mean_node.MutableInputDefs()[0], scale, bias};
    Node& layer_norm_node = graph.AddNode(graph.GenerateNodeName("LayerNormalization"),
                                          "LayerNormalization",
                                          "fused LayerNorm subgraphs ",
                                          layer_norm_input_defs,
                                          {}, {}, kOnnxDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the layer_norm_node.
    // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime