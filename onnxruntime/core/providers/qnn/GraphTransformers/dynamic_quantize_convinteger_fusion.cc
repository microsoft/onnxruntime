// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_quantize_convinteger_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

// Check if bias is a 1-D tensor, or N-D tensor with the prior N-1 dimension equal to 1.
// And its last dimension equal to Conv's last dimension
static bool CheckBiasShape(const TensorShapeProto* bias_shape) {
  if (nullptr == bias_shape || bias_shape->dim_size() < 1) {
    return false;
  }

  // First N-1 dimension must equal to 1
  for (int i = 0; i < bias_shape->dim_size() - 1; i++) {
    if (bias_shape->dim(i).dim_value() != 1) {
      return false;
    }
  }

  int64_t bias_last_dim = bias_shape->dim(bias_shape->dim_size() - 1).dim_value();

  // Don't allow last dimension to be 1, to be on the safe side
  return bias_last_dim > 1;
}

/**
DynamicQuantizeConvIntegerFusion will fuse subgraph like below into DequantizeLinear + Conv:
                (input)
                  |
                  v
          DynamicQuantizeLinear
           |       /  \
           |      /    \
B   B_Zero A A_Zero  A_Scale  B_Scale  Bias (Const, Optional)                        (B, B_Scale, B_Zero)
 \    |    |    /        \      /             |                                                |
  \   |    |   /          \    /              |                                                |
   \  |    |  /            \  /               |                                   input   DequantizeLinear
   ConvInteger              Mul               |                                      \       /
     |                       |                |                                       \     /
     v                       v                |                                        \   /
    Cast ------------------>Mul               |                     ---->              Conv
                             |                |                                          |
                             v                |                                          v
                            Add <-------------+                                       (output)
                             |
                             v
                         (output)

*/
// Status DynamicQuantizeConvIntegerFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
Status DynamicQuantizeConvIntegerFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const onnxruntime::logging::Logger& logger) const {
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  const Node* p_convinteger_node = graph.GetNode(node.Index());

  const Node* p_cast_node = graph_utils::FirstChildByType(*p_convinteger_node, "Cast");
  ORT_ENFORCE(p_cast_node);

  const Node* p_mul_node = graph_utils::FirstChildByType(*p_cast_node, "Mul");
  ORT_ENFORCE(p_mul_node);

  const Node* p_dynamicquantize_node = graph_utils::FirstParentByType(*p_convinteger_node, "DynamicQuantizeLinear");
  ORT_ENFORCE(p_dynamicquantize_node);

  const Node* p_mul_node_right = graph_utils::FirstParentByType(*p_mul_node, "Mul");
  ORT_ENFORCE(p_mul_node_right);

  Node& cast_node = *graph.GetNode(p_cast_node->Index());
  Node& convinteger_node = *graph.GetNode(p_convinteger_node->Index());
  Node& dynamicquantize_node = *graph.GetNode(p_dynamicquantize_node->Index());
  Node& mul_node_right = *graph.GetNode(p_mul_node_right->Index());
  Node& mul_node = *graph.GetNode(p_mul_node->Index());

  // Check Nodes' Edges count and Nodes' outputs are not in Graph output
  if (!optimizer_utils::CheckOutputEdges(graph, cast_node, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, convinteger_node, 1) ||
      //! optimizer_utils::CheckOutputEdges(graph, dynamicquantize_node, 3) ||
      !optimizer_utils::CheckOutputEdges(graph, mul_node_right, 1)) {
    return Status::OK();
  }

  // Find bias node
  Node* p_add_node = nullptr;
  if (optimizer_utils::CheckOutputEdges(graph, mul_node, 1)) {
    const Node* tmp_add_node = graph_utils::FirstChildByType(mul_node, "Add");
    if (nullptr != tmp_add_node) {
      const NodeArg& tmp_add_node_B = *(tmp_add_node->InputDefs()[1]);
      if (graph_utils::IsConstantInitializer(graph, tmp_add_node_B.Name(), true) &&
          CheckBiasShape(tmp_add_node_B.Shape())) {
        p_add_node = graph.GetNode(tmp_add_node->Index());
      }
    }
  }

  NodeArg optional_node_arg("", nullptr);
  InlinedVector<NodeArg*> dql_input_defs{
      convinteger_node.MutableInputDefs()[1],  // B
      mul_node_right.MutableInputDefs()[1],    // B_Scale
      &optional_node_arg};

  // Add zero point of B
  if (p_convinteger_node->InputDefs().size() >= 4) {
    dql_input_defs[2] = convinteger_node.MutableInputDefs()[3];
  }

  InlinedVector<NodeArg*> conv_input_defs{
      dynamicquantize_node.MutableInputDefs()[0],
      &optional_node_arg};

  std::string dql_op_type = "DequantizeLinear";
  auto dql_output_type_proto = *dql_input_defs[0]->TypeAsProto();
  const ONNX_NAMESPACE::TensorProto_DataType element_type =
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(conv_input_defs[0]->TypeAsProto()->tensor_type().elem_type());
  dql_output_type_proto.mutable_tensor_type()->set_elem_type(element_type);
  auto& dql_output = graph.GetOrCreateNodeArg(dql_input_defs[0]->Name() + " _dequantize", &dql_output_type_proto);
  Node& fused_node_1 = graph.AddNode(graph.GenerateNodeName(convinteger_node.Name() + "_dequantize_convinteger"),
                                     dql_op_type,
                                     "",
                                     dql_input_defs,
                                     {&dql_output},
                                     {},
                                     kOnnxDomain);
  // Assign provider to this new node. Provider should be same as the provider for old node.
  fused_node_1.SetExecutionProviderType(mul_node.GetExecutionProviderType());

  // Add Dequantized Weight
  conv_input_defs[1] = fused_node_1.MutableOutputDefs()[0];

  // Add Bias
  if (p_add_node != nullptr) {
    conv_input_defs.push_back(p_add_node->MutableInputDefs()[1]);
  }

  std::string conv_op_type = "Conv";
  Node& fused_node_2 = graph.AddNode(convinteger_node.Name(),
                                     conv_op_type,
                                     "",
                                     conv_input_defs,
                                     p_add_node != nullptr ? p_add_node->MutableOutputDefs() : mul_node.MutableOutputDefs(),
                                     &convinteger_node.GetAttributes(),
                                     kOnnxDomain);
  // Assign provider to this new node. Provider should be same as the provider for old node.
  fused_node_2.SetExecutionProviderType(mul_node.GetExecutionProviderType());

  if (graph_utils::GraphEdge::GetNodeOutputEdges(dynamicquantize_node).size() > 3) {
    for (auto output_edge : graph_utils::GraphEdge::GetNodeOutputEdges(dynamicquantize_node)) {
      if (output_edge.dst_node == convinteger_node.Index() || output_edge.dst_node == mul_node_right.Index()) {
        graph.RemoveEdge(output_edge.src_node, output_edge.dst_node, output_edge.src_arg_index, output_edge.dst_arg_index);
      }
    }
  } else {
    nodes_to_remove.push_back(dynamicquantize_node);
  }
  nodes_to_remove.push_back(convinteger_node);
  nodes_to_remove.push_back(cast_node);
  nodes_to_remove.push_back(mul_node_right);
  nodes_to_remove.push_back(mul_node);
  if (p_add_node != nullptr) {
    nodes_to_remove.push_back(*p_add_node);
  }

  if (!nodes_to_remove.empty()) {
    rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  }

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  return Status::OK();
}

bool DynamicQuantizeConvIntegerFusion::SatisfyCondition(const Graph& graph, const Node& node, const onnxruntime::logging::Logger&) const {
  if (node.OpType() != "ConvInteger") {
    return false;
  }
  const Node* p_convinteger_node = graph.GetNode(node.Index());
  const Node* p_cast_node = graph_utils::FirstChildByType(*p_convinteger_node, "Cast");
  if (p_cast_node == nullptr) {
    return false;
  }

  const Node* p_mul_node = graph_utils::FirstChildByType(*p_cast_node, "Mul");
  if (p_mul_node == nullptr) {
    return false;
  }

  const Node* p_dynamicquantize_node = graph_utils::FirstParentByType(*p_convinteger_node, "DynamicQuantizeLinear");
  if (p_dynamicquantize_node == nullptr) {
    return false;
  }

  const Node* p_mul_node_right = graph_utils::FirstParentByType(*p_mul_node, "Mul");
  if (p_mul_node_right == nullptr) {
    return false;
  }

  return true;
}
}  // namespace onnxruntime
