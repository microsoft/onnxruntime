// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static bool CheckConstantInput(const Graph& graph, const NodeArg& input_arg, float expected_value) {
  auto shape = input_arg.Shape();
  if (shape == nullptr) {
    // shape inferencing wasn't able to populate shape information for this NodeArg
    return false;
  }

  auto dim_size = shape->dim_size();
  if (dim_size != 0) {
    // only check scalar.
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  if (tensor_proto == nullptr) {
    return false;
  }

  auto init_const = onnxruntime::make_unique<Initializer>(tensor_proto);
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    float* val = init_const->data<float>();
    float diff = std::abs(val[0] - static_cast<float>(expected_value));
    if (diff > FLT_EPSILON) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    double* val = init_const->data<double>();
    double diff = std::abs(val[0] - static_cast<double>(expected_value));
    if (diff > DBL_EPSILON) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    MLFloat16* val = init_const->data<MLFloat16>();
    float diff = std::abs(math::halfToFloat(val[0].val) - static_cast<float>(expected_value));
    if (diff > FLT_EPSILON) {
      return false;
    }
  }

  return true;
}

// Gelu supports limited data types.
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

Status GeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;

  for (auto node_index : node_topology_list) {
    auto& div = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(div, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div, "Div", {7}) ||
        !graph_utils::IsSupportedProvider(div, GetCompatibleExecutionProviders()) ||
        div.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(div)) {
      continue;
    }

    // Check second input is sqrt(2)
    if (!CheckConstantInput(graph, *(div.MutableInputDefs()[1]), static_cast<float>(M_SQRT2))) {
      continue;
    }

    const Node& erf_node = *(div.OutputNodesBegin());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(erf_node, "Erf", {9}) ||
        erf_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        erf_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(erf_node)) {
      continue;
    }

    const Node& add_node = *(erf_node.OutputNodesBegin());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
        add_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        add_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(add_node)) {
      continue;
    }

    // Check the other input node(e.g. not of type Erf) is 1.0f.
    const Node& add_first_input_node = *(add_node.InputNodesBegin());
    int add_const_input_index = 0;
    if (add_first_input_node.OpType().compare("Erf") == 0) {
      add_const_input_index = 1;
    }
    const auto& add_const_input_arg = add_node.InputDefs()[add_const_input_index];
    if (!CheckConstantInput(graph, *add_const_input_arg, 1.0f)) {
      continue;
    }

    const Node& mul_node = *(add_node.OutputNodesBegin());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        mul_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }

    const Node* mul2_node = nullptr;
    for (auto iter = mul_node.InputNodesBegin(); iter != mul_node.InputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Mul") == 0) {
        // find the other input node of Mul
        mul2_node = &(*iter);
        break;
      }
    }
    if (mul2_node == nullptr) {
      continue;
    }

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*mul2_node, "Mul", {7}) ||
        mul2_node->GetExecutionProviderType() != div.GetExecutionProviderType() ||
        mul2_node->GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(*mul2_node)) {
      continue;
    }

    // Check the other input node(e.g. not of type Add) is 0.5f.
    int mul_const_input_index = 0;
    if (mul2_node->InputDefs()[0]->Name() == div.MutableInputDefs()[0]->Name()) {
      mul_const_input_index = 1;
    }

    const auto& mul_const_input_arg = mul2_node->InputDefs()[mul_const_input_index];
    if (!CheckConstantInput(graph, *mul_const_input_arg, 0.5f)) {
      continue;
    }

    const std::vector<NodeArg*> gelu_input_defs{div.MutableInputDefs()[0]};
    const std::vector<NodeArg*> gelu_output_defs{const_cast<NodeArg*>(mul_node.OutputDefs()[0])};
    Node& gelu_node = graph.AddNode(graph.GenerateNodeName("Gelu"),
                                    "Gelu",
                                    "fused Gelu subgraphs ",
                                    gelu_input_defs,
                                    gelu_output_defs, {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_node.SetExecutionProviderType(div.GetExecutionProviderType());

    removed_nodes.push_front(div.Index());
    removed_nodes.push_front(erf_node.Index());
    removed_nodes.push_front(add_node.Index());
    removed_nodes.push_front(mul2_node->Index());
    removed_nodes.push_front(mul_node.Index());
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
