// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/split_quickgelu_fusion.h"

#include <deque>

#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

// To be removed
// #include <signal.h>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnxruntime;

namespace {

// pattern match on split, quickgelu and mult subgraph
// bool TrySplitQuickGeluMatch(Graph& graph, Node& start, Node*& split, Node*& quickgelu, Node*& mult, const logging::Logger& logger) {
bool TrySplitQuickGeluMatch(Graph& graph, Node& start, Node*& split, Node*& quickgelu, Node*& mult) {
  Node& node = start;
  split = quickgelu = mult = nullptr;

  // check node is split and has two outputs
  // TODO: 1. Check ONNX Op Types to Support
  // Split version 13 has axis as attribute and split as input (Should we only specify it for v13?)
  // raise(SIGTRAP);
  if (!graph_utils::IsSupportedProvider(node, {kCudaExecutionProvider, kRocmExecutionProvider})) {
    std::cout << "not cuda rocm, it is:" << node.GetExecutionProviderType() << " is it here?" << std::endl;
  }
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Split", {13, 18}) ||
      // !graph_utils::IsSupportedProvider(node, {kCudaExecutionProvider, kRocmExecutionProvider}) ||
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
  // Which operators to support?
  auto type_allowed = [](NodeArg* input) {
    auto data_type = input->TypeAsProto()->tensor_type().elem_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
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
  // What does the 0,0 represent here?
  // node -> getconsumer

  // check all output edges
  const Node* p_quickgelu_node = nullptr;
  const Node* p_mul_node = nullptr;
  unsigned int quickgelu_count = 0;
  unsigned int mul_count = 0;
  unsigned int other_count = 0;
  for (auto it = split_node.OutputNodesBegin(); it != split_node.OutputNodesEnd(); ++it) {
    if ((*it).OpType().compare("QuickGelu") == 0) {
      p_quickgelu_node = &(*it);
      quickgelu_count++;
    } else if ((*it).OpType().compare("Mul") == 0) {
      p_mul_node = &(*it);
      mul_count++;
    } else {
      other_count++;
    }
  }

  // QuickGelu and Mul Node count should exactly be 1, other nodes should be 0
  if (quickgelu_count != 1 || mul_count != 1 || other_count != 0) {
    return false;
  }
  Node& quickgelu_node = *graph.GetNode(p_quickgelu_node->Index());
  Node& mul_node = *graph.GetNode(p_mul_node->Index());

  // Check Mul node is also the output of QuickGelu
  Node& next_node = *graph.GetNode(quickgelu_node.OutputNodesBegin()->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Mul", {7, 13, 14})) {
    return false;
  }

  // pattern match succeeded
  split = &split_node;
  quickgelu = &quickgelu_node;
  mult = &mul_node;
  return true;
}

// get parameters
bool GetSplitQuickGeluParams(
    Node& split_node,
    Node& quickgelu_node,
    NodeArg*& input,
    int& axis,
    float& alpha) {
  input = split_node.MutableInputDefs()[0];
  axis = -1;
  alpha = -1.0;
  auto& split_attr = split_node.GetAttributes();
  if (split_attr.find("axis") != split_attr.end()) {
    auto& axis_attr = split_attr.at("axis");
    axis = utils::HasInt(axis_attr) ? (int)axis_attr.i() : axis;
  } else {
    return false;
  }
  auto& quickgelu_attr = quickgelu_node.GetAttributes();
  // for (const auto& pair : quickgelu_attr){
  //   std::cout << "Key:" << pair.first << std::endl;
  // }
  if (quickgelu_attr.find("alpha") != quickgelu_attr.end()) {
    auto& alpha_attr = quickgelu_attr.at("alpha");
    alpha = utils::HasFloat(alpha_attr) ? (float)alpha_attr.f() : alpha;
    // printf("Got ALPHA Value as: %f\n", alpha);
  } else {
    return false;
  }
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
    float alpha) {
  std::string fused_desc =
      "fused " + split_node.Name() + " and " + quickgelu_node.Name() + " and " + mul_node.Name() + " into SplitQuickGelu";

  std::string op_type = "S2SModelSplitQuickGelu";
  Node& fused_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                   op_type,
                                   fused_desc,
                                   std::array{input},
                                   {},
                                   {},
                                   kMSDomain);

  // add split axis
  // add QuickGelu alpha
  fused_node.AddAttribute("axis", static_cast<int64_t>(axis));
  fused_node.AddAttribute("alpha", static_cast<float>(alpha));

  // finalize node fusion (e.g. remove old nodes and shift outputs)
  fused_node.SetExecutionProviderType(split_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, {split_node, quickgelu_node, mul_node}, fused_node);
}

}  // namespace

namespace onnxruntime {

Status SplitQuickGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // only support GPU execution provider
  auto& cep = GetCompatibleExecutionProviders();
  if (cep.size() > 0 && cep.find(kCudaExecutionProvider) == cep.end() && cep.find(kRocmExecutionProvider) == cep.end())
    return Status::OK();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    Node *split_node, *quickgelu_node, *mul_node;
    // if (!TrySplitQuickGeluMatch(graph, node, split_node, quickgelu_node, mul_node, logger)) {
    if (!TrySplitQuickGeluMatch(graph, node, split_node, quickgelu_node, mul_node)) {
      continue;
    }

    NodeArg* input;
    int axis;
    float alpha;
    // Call this from match fn
    if (!GetSplitQuickGeluParams(*split_node, *quickgelu_node, input, axis, alpha)) {
      continue;
    }
    FuseSplitQuickGeluSubgraph(graph, *split_node, *quickgelu_node, *mul_node, input, axis, alpha);
    modified = true;
    std::cout << "FUSION SUBGRAPH COMPLETE NOW" << std::endl;

    VLOGF(logger, 1, "Fused S2S Model Split + QuickGelu into S2SModelSplitQuickGelu node.\n");
  }

  return Status::OK();
}

}  // namespace onnxruntime
