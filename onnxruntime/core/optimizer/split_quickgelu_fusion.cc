// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/split_quickgelu_fusion.h"

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnxruntime;

namespace {

// pattern match on split, quickgelu and mult subgraph
bool TrySplitQuickGeluMatch(Graph& graph, Node& start, Node*& split, Node*& quickgelu, Node*& mult) {
  Node& node = start;
  add = quickgelu = mult = nullptr;

  // check node is split and has two outputs
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Split", {14}) ||
      !graph_utils::IsSupportedProvider(node, {kCudaExecutionProvider, kRocmExecutionProvider}) ||
      !optimizer_utils::CheckOutputEdges(graph, node, 2)) {
    return false;
  }

  // check shape information is not available for input
  Node& split_node = node;
  NodeArg* input = split_node.MutableInputDefs()[0];
  const TensorShapeProto* S = input->Shape();
  if (S == nullptr || S->dim_size() < 1) {
    return false;
  }

  // Split supports only float/float16/double/bfloat16 (OR MORE!!!!!????) - see ./onnxruntime/core/graph/contrib_ops/contrib_defs.cc
  auto type_allowed = [](NodeArg* input) {
    auto data_type = input->TypeAsProto()->tensor_type().elem_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE &&
        data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
        data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        data_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      return false;
    }
    return true;
  };
  if (!type_allowed(input)) {
    return false;
  }

  // Trying to find Split->QuickGelu->Mul Path
  std::vector<const Node::EdgeEnd*> edges;
  std::vector<graph_utils::EdgeEndToMatch> quickgelu_mul_path{
      {0, 0, "QuickGelu", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Mul", {7, 13, 14}, kOnnxDomain}};

  if (!graph_utils::FindPath(node, true, quickgelu_mul_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for QuickGelu mul operation.");
    return false;
  }
  for (size_t i = 0; i < edges.size(); i++) {
    if (!optimizer_utils::CheckOutputEdges(graph, edges[i]->GetNode(), 1)) {
      DEBUG_LOG("Output edge count not expected for nodes.");
      return false;
    }
  }
  Node& quickgelu_node = *graph.GetNode(edges[0]->GetNode().Index());
  Node& mul_node = *graph.GetNode(edges[1]->GetNode().Index());

  std::vector<graph_utils::EdgeEndToMatch> only_mul_path{
      {0, 0, "Mul", {7, 13, 14}, kOnnxDomain}};

  if (!graph_utils::FindPath(node, true, only_mul_path, edges, logger)) {
    DEBUG_LOG("Failed to find for direct Mul.");
    return false;
  }
  for (size_t i = 0; i < edges.size(); i++) {
    if (!optimizer_utils::CheckOutputEdges(graph, edges[i]->GetNode(), 1)) {
      DEBUG_LOG("Output edge count not expected for nodes.");
      return false;
    }
  }

  Node& mul_node_2 = *graph.GetNode(edges[0]->GetNode().Index());

  // Compare if the two mul_nodes are same
  // Figure this out?


  // pattern match succeeded
  split = &split_node;
  quickgelu = &quickgelu_node;
  mult = &mul_node;
  return true;
}

// coalesce subgraph nodes into fused node
void FuseSplitQuickGeluSubgraph(
    Graph& graph,
    Node& split_node,
    Node& quickgelu_node,
    Node& mul_node,
    NodeArg* input,
    int axis,
    int alpha) {
  std::string fused_desc =
      "fused " + split_node.Name() + " and " + quickgelu_node.Name() + " and " + mul.Name() + " into SplitQuickGelu";

  std::string op_type = "SplitQuickGelu";
  Node& fused_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                   op_type,
                                   fused_desc,
                                   input,
                                   {},
                                   {},
                                   kMSDomain);

  // add split axis
  // add QuickGelu alpha
  fused_node.AddAttribute("axis", static_cast<int64_t>(axis));
  fused_node.AddAttribute("alpha", static_cast<int64_t>(alpha));

  // finalize node fusion (e.g. remove old nodes and shift outputs)
  fused_node.SetExecutionProviderType(split_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, {split_node, quickgelu_node, mul_node}, fused_node);
}

}  // namespace

bool CastElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  const auto* input_type = node.InputDefs()[0]->TypeAsProto();
  if (input_type == nullptr || !input_type->tensor_type().has_elem_type()) {
    return false;
  }

  return optimizer_utils::IsAttributeWithExpectedValue(node, "to", static_cast<int64_t>(input_type->tensor_type().elem_type()));
}

}  // namespace onnxruntime
