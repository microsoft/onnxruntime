// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_bn_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace matmulbnfusion {
  std::vector<std::pair<std::string, InlinedVector<ONNX_NAMESPACE::OperatorSetVersion>>> ignorable_nodes{
    {"Reshape", {1, 5, 13, 14, 19}},
    {"Transpose", {1, 13}}};
  std::pair<std::string, InlinedVector<ONNX_NAMESPACE::OperatorSetVersion>> dest = {"BatchNormalization", {1, 6, 7, 9, 14, 15}};
}

std::optional<std::reference_wrapper<const Node>> MatchPath(
  const Node& parent_node, 
  const Node& curr_node,
  const std::pair<std::string, InlinedVector<ONNX_NAMESPACE::OperatorSetVersion>>& dest, 
  const gsl::span<std::pair<std::string, InlinedVector<ONNX_NAMESPACE::OperatorSetVersion>>>& ignorable_nodes,
  std::vector<bool>& ignorable_nodes_visited) {
  
  // curr_node has different execution provider then it's parent or has > 1 output
  if (curr_node.GetExecutionProviderType() != parent_node.GetExecutionProviderType() ||
      curr_node.GetOutputEdgesCount() != 1) {
    return std::nullopt;
  }

  // curr_node == dest_node
  if (graph_utils::IsSupportedOptypeVersionAndDomain(curr_node, dest.first, dest.second)) {
    return curr_node;
  }

  // curr_node can be any of the ignorable_nodes.
  for (size_t index = 0; index < ignorable_nodes.size(); index++) {
    if (!ignorable_nodes_visited[index] &&
        graph_utils::IsSupportedOptypeVersionAndDomain(curr_node, ignorable_nodes[index].first, ignorable_nodes[index].second)) {
      ignorable_nodes_visited[index] = true;
      return MatchPath(curr_node, *curr_node.OutputNodesBegin(), dest, ignorable_nodes, ignorable_nodes_visited);
    }
  }

  // curr_node neither a dest node nor any of the ignorable_nodes.
  return std::nullopt;
}

/*
 *   Given a MatMul node, it will verify the following pattern.
 *                MatMul                  GEMM 
 *                  |                       |     
 *               Reshape ^     --->      Reshape ^
 *                  |                       |
 *             Transpose ^             Transpose ^
 *                  |
 *        BatchNormalization
 * Note: ^ means there can be 0 or 1 occurrences of that node. 
 * Other Conditions:
 *   - B tensor of MatMul should be constant.
 *   - scale, B, mean, var tensors of BatchNormalization should be constant.
 *   - Every node in the path, except the BatchNormalization, should have only 1 output edge.
 */
bool MatmulBNFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {1, 9, 13}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const Node& child_node = *node.OutputNodesBegin();
  std::vector<bool> ignorable_nodes_visited(matmulbnfusion::ignorable_nodes.size(), false);
  std::optional<std::reference_wrapper<const Node>> batch_norm_node = MatchPath(
    node,
    child_node,
    matmulbnfusion::dest,
    matmulbnfusion::ignorable_nodes,
    ignorable_nodes_visited);
  if (!batch_norm_node.has_value()) {
    return false;
  }

  // Check that the appropriate inputs to the Matmul and BN nodes are constants.
  if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1]) ||
      !graph_utils::NodeArgIsConstant(graph, *batch_norm_node->get().InputDefs()[1]) ||
      !graph_utils::NodeArgIsConstant(graph, *batch_norm_node->get().InputDefs()[2]) ||
      !graph_utils::NodeArgIsConstant(graph, *batch_norm_node->get().InputDefs()[3]) ||
      !graph_utils::NodeArgIsConstant(graph, *batch_norm_node->get().InputDefs()[4])) {
    return false;
  }

  // First output from BN is required. Others are optional. If any optional outputs exist we can't fuse.
  const auto& output_defs = batch_norm_node->get().OutputDefs();
  if (output_defs.size() > 1) {
    for (size_t i = 1, end = output_defs.size(); i < end; ++i) {
      if (output_defs[i] != nullptr && output_defs[i]->Exists()) {
        return false;
      }
    }
  }

  if (graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  return true;
}

/*
 * BatchNormalization: [https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_batch_normalization_operator_desc]
 *   Scale * ((Input - Mean) / sqrt(Variance + Epsilon)) + Bias // ignore the FusedActivation in the above definition, that's very specific to DML
 * Expanding out the terms:
 *   Output = (Scale / sqrt(Variance + Epsilon)) * Input + (Scale / sqrt(Variance + Epsilon)) * -Mean + Bias
 * Here,
 *   [Scale/sqrt(Variance + Epsilon)] is constant, and let's call it `alpha`
 *   [(Scale / sqrt(Variance + Epsilon)) * -Mean + Bias] is also constant, and let's call it `beta`
 * Output = alpha * Input + beta, Input = B tensor of MatMul.
 *
 */
Status MatmulBNFusion::Apply(Graph& graph, Node& matmul_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const Node& child_node = *matmul_node.OutputNodesBegin();
  std::vector<bool> ignorable_nodes_visited(matmulbnfusion::ignorable_nodes.size(), false);
  NodeIndex batch_norm_node_index = MatchPath(
      matmul_node,
      child_node,
      matmulbnfusion::dest,
      matmulbnfusion::ignorable_nodes,
      ignorable_nodes_visited)->get().Index();

  Node& batch_norm_node = *graph.GetNode(batch_norm_node_index); // need mutable node, that's why extracting node from graph

  // only perform fusion if epsilon is present and is of float_32 type
  auto epsilon_attribute = batch_norm_node.GetAttributes().find("epsilon");
  if (epsilon_attribute == batch_norm_node.GetAttributes().end() ||
      epsilon_attribute->second.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT) {
    return Status::OK();
  }
  const float epsilon = epsilon_attribute->second.f();

  const onnx::TensorProto* scale_tensor = graph_utils::GetConstantInitializer(graph, batch_norm_node.InputDefs()[1]->Name());
  ORT_ENFORCE(scale_tensor);
  const onnx::TensorProto* bias_tensor = graph_utils::GetConstantInitializer(graph, batch_norm_node.InputDefs()[2]->Name());
  ORT_ENFORCE(bias_tensor);
  const onnx::TensorProto* mean_tensor = graph_utils::GetConstantInitializer(graph, batch_norm_node.InputDefs()[3]->Name());
  ORT_ENFORCE(mean_tensor);
  const onnx::TensorProto* var_tensor = graph_utils::GetConstantInitializer(graph, batch_norm_node.InputDefs()[4]->Name());
  ORT_ENFORCE(var_tensor);
  const onnx::TensorProto* matmul_b_tensor = graph_utils::GetConstantInitializer(graph, matmul_node.InputDefs()[1]->Name());
  ORT_ENFORCE(matmul_b_tensor);

  if (!optimizer_utils::IsFloatingPointDataType(*matmul_b_tensor) ||
      !optimizer_utils::IsFloatingPointDataType(*scale_tensor) ||
      !optimizer_utils::IsFloatingPointDataType(*bias_tensor) ||
      !optimizer_utils::IsFloatingPointDataType(*mean_tensor) ||
      !optimizer_utils::IsFloatingPointDataType(*var_tensor) ||
      scale_tensor->dims_size() != 1 ||
      bias_tensor->dims_size() != 1 ||
      mean_tensor->dims_size() != 1 ||
      var_tensor->dims_size() != 1 ||
      scale_tensor->dims(0) != matmul_b_tensor->dims(1) ||
      bias_tensor->dims(0) != matmul_b_tensor->dims(1) ||
      mean_tensor->dims(0) != matmul_b_tensor->dims(1) ||
      var_tensor->dims(0) != matmul_b_tensor->dims(1)) {
    return Status::OK();
  }

  /*
   * temp = scale / sqrt(var + epsilon)
   * output = (temp * Input) - ((temp * mean) + bias)
   */
  Initializer scale(*scale_tensor, graph.ModelPath());
  Initializer bias(*bias_tensor, graph.ModelPath());
  Initializer mean(*mean_tensor, graph.ModelPath());
  Initializer var(*var_tensor, graph.ModelPath());
  Initializer matmul_b(*matmul_b_tensor, graph.ModelPath());

  var.add(epsilon);
  var.sqrt();
  scale.div(var);  // this is the temp
  matmul_b.scale_by_axis(scale, 1, true);

  mean.mul(scale);
  bias.sub(mean);

  // create B tensorProto for new Gemm node from <matmulB> initializer.
  ONNX_NAMESPACE::TensorProto new_gemm_b_tensor(*matmul_b_tensor);
  matmul_b.ToProto(new_gemm_b_tensor);
  const std::string new_gemm_b_name = graph.GenerateNodeArgName("MatMulBnFusion_GemmB_" + matmul_b_tensor->name());
  new_gemm_b_tensor.set_name(new_gemm_b_name);
  NodeArg& new_gemm_b_node_arg = graph_utils::AddInitializer(graph, new_gemm_b_tensor);

  // create bias tensorProto for new Gemm node from <bias> initializer.
  ONNX_NAMESPACE::TensorProto new_gemm_bias_tensor(*bias_tensor);
  bias.ToProto(new_gemm_bias_tensor);
  const std::string new_gemm_bias_name = graph.GenerateNodeArgName("MatMulBnFusion_GemmBias");
  new_gemm_bias_tensor.set_name(new_gemm_bias_name);
  NodeArg& new_gemm_bias_node_arg = graph_utils::AddInitializer(graph, new_gemm_bias_tensor);

  Node& gemm_node = graph.AddNode(
      graph.GenerateNodeArgName("MatMulBnFusion_Gemm"),
      "Gemm",
      "Generated from Matmul BatchNormalization fusion",
      {matmul_node.MutableInputDefs()[0], &new_gemm_b_node_arg, &new_gemm_bias_node_arg},
      matmul_node.MutableOutputDefs(),
      nullptr,
      kOnnxDomain);

  // Remove MatMul node.
  Node* node = graph.GetNode(matmul_node.Index());
  graph_utils::RemoveNodeOutputEdges(graph, *node);
  graph.RemoveNode(matmul_node.Index());

  // Delete optional empty output defs.
  // Delete BatchNormalization node and update the input of the child of BatchNormalization
  batch_norm_node.MutableOutputDefs().resize(1);
  NodeIndex batch_norm_parent_index = child_node.OpType() == "BatchNormalization" ? gemm_node.Index() :
    batch_norm_node.InputNodesBegin()->Index();
  graph_utils::FinalizeNodeFusion(graph, *graph.GetNode(batch_norm_parent_index), batch_norm_node);

  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}
}  // namespace onnxruntime