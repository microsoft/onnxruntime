// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(double)"};
// Default epsilon
static const float DEFAULT_LAYERNORM_EPSILON = 1e-5f;

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

In recent pytorch, Cast nodes may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type 
due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                                        ^
                      |                                                        |
                      +--------------------------------------------------------+
*/
Status LayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_reduce_mean = graph.GetNode(node_index);
    if (p_reduce_mean == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reduce_mean_node = *p_reduce_mean;
    ORT_RETURN_IF_ERROR(Recurse(reduce_mean_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_node, "ReduceMean", {1, 11, 13}) ||
        !graph_utils::IsSupportedProvider(reduce_mean_node, GetCompatibleExecutionProviders()) ||
        (reduce_mean_node.GetOutputEdgesCount() != 1 && reduce_mean_node.GetOutputEdgesCount() != 2) ||
        !graph.GetNodeOutputsInGraphOutputs(reduce_mean_node).empty() ||
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
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node, "Sub", {7, 13}) ||
        sub_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sub_node, subCnt == 1 ? 2u : 1u) ||
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
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node_dup, "Sub", {7, 13}) ||
          sub_node_dup.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, sub_node, 1) ||
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
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7, 13}) ||
        div_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, div_node, 1) ||
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

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6, 13}) ||
        sqrt_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1) ||
        !IsSupportedDataType(sqrt_node) ||
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(sqrt_node);

    // Traceback the sqrt node to find add --> sqrt
    Node& add2_node = *graph.GetNode(sqrt_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7, 13}) ||
        add2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, add2_node, 1) ||
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
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean2_node, "ReduceMean", {1, 11, 13}) ||
        reduce_mean2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, reduce_mean2_node, 1) ||
        !IsSupportedDataType(reduce_mean2_node) ||
        reduce_mean2_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean2_node);

    // Traceback the reduceMean node to find pow --> reduceMean
    Node& pow_node = *graph.GetNode(reduce_mean2_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7, 12, 13}) ||
        pow_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, pow_node, 1) ||
        !IsSupportedDataType(pow_node)) {
      continue;
    }
    nodes_to_remove.push_back(pow_node);

    // check if Cast node exists between sub and pow
    const Node* p_cast_node = graph_utils::FirstParentByType(pow_node, "Cast");
    if (p_cast_node != nullptr) {
      Node& cast_node = *graph.GetNode(pow_node.InputNodesBegin()->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9, 13}) ||
          cast_node.GetExecutionProviderType() != cast_node.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, cast_node, 1) ||
          !IsSupportedDataType(cast_node)) {
        continue;
      }
      nodes_to_remove.push_back(cast_node);
    }

    // Traceback from the last node in vector to find sub --> pow  or  sub --> cast
    const Node* p_sub2_node = graph_utils::FirstParentByType(nodes_to_remove.back(), "Sub");
    if (p_sub2_node == nullptr ||
        (p_sub2_node != p_sub_node && p_sub2_node != p_sub_node_dup)) {
      continue;
    }

    // div --> mul
    Node& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13}) ||
        mul_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, mul_node, 1) ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }
    nodes_to_remove.push_back(mul_node);

    // mul --> add
    // Need not check output edges of last node since they will be moved to fused node.
    Node& last_add_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(last_add_node, "Add", {7, 13}) ||
        last_add_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(last_add_node)) {
      continue;
    }
    nodes_to_remove.push_back(last_add_node);

    // get axes attributes
    const onnxruntime::NodeAttributes& attributes = reduce_mean_node.GetAttributes();
    std::vector<int64_t> axes_values;
    if (attributes.find("axes") != attributes.end()) {
      axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    }

    // Get the inputs for the new LayerNormalization node.
    // scale and bias could be multi-dims; we only support it for training at the moment
    // because SkipLayerNorm kernel, for example, has dependency on single dim size
    NodeArg* scale = nullptr;
    NodeArg* bias = nullptr;
    for (size_t i = 0; i < mul_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(mul_node.MutableInputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, mul_node.MutableInputDefs()[i])) {
#ifdef ENABLE_TRAINING
        if (axes_values.empty() ||
            mul_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
          scale = mul_node.MutableInputDefs()[i];
        }
#else
        // Scale must be 1d.
        if (mul_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          scale = mul_node.MutableInputDefs()[i];
        }
#endif
      }
    }

    for (size_t i = 0; i < last_add_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(last_add_node.MutableInputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, last_add_node.MutableInputDefs()[i])) {
#ifdef ENABLE_TRAINING
        if (axes_values.empty() ||
            last_add_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
          bias = last_add_node.MutableInputDefs()[i];
        }
#else
        // Bias must be 1d.
        if (last_add_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          bias = last_add_node.MutableInputDefs()[i];
        }
#endif
      }
    }
    if (scale == nullptr || bias == nullptr) {
      continue;
    }

    // Scale and bias must have the same shape.
    bool same_dim = true;
    for (int i = 0; i < scale->Shape()->dim_size(); i++) {
      if (scale->Shape()->dim(i).dim_value() != bias->Shape()->dim(i).dim_value()) {
        same_dim = false;
        break;
      }
    }
    if (!same_dim)
      continue;

    const std::vector<NodeArg*> layer_norm_input_defs{reduce_mean_node.MutableInputDefs()[0], scale, bias};
    Node& layer_norm_node = graph.AddNode(graph.GenerateNodeName("LayerNormalization"),
                                          "LayerNormalization",
                                          "fused LayerNorm subgraphs ",
                                          layer_norm_input_defs,
                                          {}, {}, kOnnxDomain);

    // Get constant "epsilon" from "Add2" node if available. Else, default value will be used.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, add2_node.MutableInputDefs()[1]->Name());
    if (tensor_proto != nullptr &&
        tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      Initializer initializer{*tensor_proto, graph.ModelPath()};
      layer_norm_node.AddAttribute("epsilon", initializer.data<float>()[0]);
    } else {
      layer_norm_node.AddAttribute("epsilon", DEFAULT_LAYERNORM_EPSILON);
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the layer_norm_node.
    // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);

#ifdef ENABLE_TRAINING
    // add two extra output defs, so we have 3 output defs that match what gradient builder expected
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_mean"), nullptr));
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std_var"), nullptr));
#endif

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime
