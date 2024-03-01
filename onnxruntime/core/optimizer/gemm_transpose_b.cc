// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gemm_transpose_b.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status GemmTransposeB::Apply(Graph& graph, Node& node, RewriteRuleEffect& modified, const logging::Logger&) const {
  auto& gemm_node = node;
  const auto& gemm_inputs = gemm_node.InputDefs();

  const auto& B_input_name = gemm_inputs[1]->Name();
  const auto* gemm_b_tensor_proto = graph_utils::GetConstantInitializer(graph, B_input_name);
  ORT_ENFORCE(gemm_b_tensor_proto);

  // Conv only supports floating point data types, so can only fuse with an initializer containing those types
  if (!optimizer_utils::IsFloatingPointDataType(*gemm_b_tensor_proto) ||
      gemm_b_tensor_proto->dims_size() != 2) {
    return Status::OK();
  }

  Initializer gemm_B{*gemm_b_tensor_proto, graph.ModelPath()};
  gemm_B.transpose();

  // Create new initializers of conv
  ONNX_NAMESPACE::TensorProto new_gemm_B_tensor_proto;
  gemm_B.ToProto(new_gemm_B_tensor_proto);

  auto new_B_name = graph.GenerateNodeArgName("GemmTransposeB_" + B_input_name);
  new_gemm_B_tensor_proto.set_name(new_B_name);

  NodeArg& new_gemm_B_node_arg = graph_utils::AddInitializer(graph, new_gemm_B_tensor_proto);
  graph_utils::ReplaceNodeInput(node, 1, new_gemm_B_node_arg);

  gemm_node.ClearAttribute("transB");

  modified = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

bool GemmTransposeB::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {1, 6, 7, 9, 11, 13}) ||
      node.GetOutputEdgesCount() > 1) {
    return false;
  }

  // get currently set attributes of Gemm
  bool transB = static_cast<bool>(node.GetAttributes().at("transB").i());
  if (!transB) {
    return false;
  }

  // Check that B is constant.
  if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1])) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
