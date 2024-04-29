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
bool TrySplitQuickGeluMatch(Graph& graph, Node& start, Node*& split, Node*& quickgelu, Node*& mult, const logging::Logger& logger) {
// bool TrySplitQuickGeluMatch(Graph& graph, Node& start, Node*& split, Node*& quickgelu, Node*& mult) {
  Node& node = start;
  split = quickgelu = mult = nullptr;

  // check node is split and has two outputs
  // TODO: 1. Check ONNX Op Types to Support
  // Split version 13 has axis as attribute and split as input (Should we only specify it for v13?)
  // raise(SIGTRAP);
  std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!0" << std::endl;
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Split", {11, 13, 18})) {
    std::cout << "not op type 11, 13, 18" << std::endl;
  }
  if (!graph_utils::IsSupportedProvider(node, {kCudaExecutionProvider, kRocmExecutionProvider})) {
    std::cout << "not cuda rocm, it is:" << node.GetExecutionProviderType() << " is it here?" << std::endl;
  }
  if (!optimizer_utils::CheckOutputEdges(graph, node, 2)) {
    std::cout << "not output edges 2" << std::endl;
  }
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Split", {11, 13, 18}) ||
      // !graph_utils::IsSupportedProvider(node, {kCudaExecutionProvider, kRocmExecutionProvider}) ||
      !optimizer_utils::CheckOutputEdges(graph, node, 2)) {
    return false;
  }
  std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1" << std::endl;

  // check shape information is not available for input
  Node& split_node = node;
  NodeArg* input = split_node.MutableInputDefs()[0];
  const TensorShapeProto* S = input->Shape();
  if (S == nullptr || S->dim_size() < 1) {
    return false;
  }
  std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2" << std::endl;

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
  std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3" << std::endl;
  if (!type_allowed(input)) {
    return false;
  }
  std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!4" << std::endl;

  // Trying to find Split->QuickGelu->Mul Path
  // What does the 0,0 represent here?
  // node -> getconsumer
  //

  // check add is only consumed by softmax with matching exec provider
  Node& next_node = *graph.GetNode(split_node.OutputNodesBegin()->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "QuickGelu", {1}, kMSDomain)) {
    std::cout << "not QuickGelu" << std::endl;
    return false;
  }
  Node& quickgelu_node = next_node;
  Node& quickgelu_next_node = *graph.GetNode(quickgelu_node.OutputNodesBegin()->Index());

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(quickgelu_next_node, "Mul", {7, 13, 14})) {
    std::cout << "not Mul Node" << std::endl;
    return false;
  }
  Node& mul_node = quickgelu_next_node;
  if (next_node.GetExecutionProviderType() != split_node.GetExecutionProviderType()) {
    std::cout << "Mismatch EP Type" << std::endl;
  }

  std::vector<const Node::EdgeEnd*> edges;
  std::vector<graph_utils::EdgeEndToMatch> quickgelu_path{
    {0, 0, "QuickGelu", {1}, kMSDomain}};

  if (!graph_utils::FindPath(node, true, quickgelu_path, edges, logger)) {
    std::cout << "Failed to find path for QuickGelu operation." << std::endl;
    DEBUG_LOG("Failed to find path for QuickGelu operation.");
    return false;
  }



  // std::vector<const Node::EdgeEnd*> edges;
  // // TODO: Replace QuickGelu by other Elementwise Op for better generalization
  // std::vector<graph_utils::EdgeEndToMatch> quickgelu_mul_path{
  //     {0, 0, "QuickGelu", {1}, kMSDomain},
  //     {0, 0, "Mul", {7, 13, 14}, kOnnxDomain}};

  // if (!graph_utils::FindPath(node, true, quickgelu_mul_path, edges, logger)) {
  //   DEBUG_LOG("Failed to find path for QuickGelu mul operation.");
  //   return false;
  // }
  std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!5" << std::endl;
  // for (size_t i = 0; i < edges.size(); i++) {
  //   if (!optimizer_utils::CheckOutputEdges(graph, edges[i]->GetNode(), 1)) {
  //     DEBUG_LOG("Output edge count not expected for nodes.");
  //     return false;
  //   }
  // }
  // std::cout << "Continuing part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!6" << std::endl;
  // Node& quickgelu_node = *graph.GetNode(edges[0]->GetNode().Index());
  // Node& mul_node = *graph.GetNode(edges[1]->GetNode().Index());

  // std::vector<graph_utils::EdgeEndToMatch> only_mul_path{
  //     {0, 0, "Mul", {7, 13, 14}, kOnnxDomain}};

  // if (!graph_utils::FindPath(node, true, only_mul_path, edges, logger)) {
  //   DEBUG_LOG("Failed to find for direct Mul.");
  //   return false;
  // }
  // for (size_t i = 0; i < edges.size(); i++) {
  //   if (!optimizer_utils::CheckOutputEdges(graph, edges[i]->GetNode(), 1)) {
  //     DEBUG_LOG("Output edge count not expected for nodes.");
  //     return false;
  //   }
  // }

  // Node& mul_node_2 = *graph.GetNode(edges[0]->GetNode().Index());

  // Compare if the two mul_nodes are same
  // Figure this out?
  // node api to get name and then compare

  // pattern match succeeded
  split = &split_node;
  quickgelu = &quickgelu_node;
  mult = &mul_node;
  std::cout << "FINISHED MATCH, RETURNING TRUE" << std::endl;
  return true;
}

// get parameters
bool GetSplitQuickGeluParams(
    Node& split_node,
    Node& quickgelu_node,
    NodeArg*& input,
    int& axis,
    int& alpha) {
  std::cout << "Params part 1" << std::endl;
  input = split_node.MutableInputDefs()[0];
  axis = -1;
  alpha = -1;
  std::cout << "Params part 2" << std::endl;
  auto& split_attr = split_node.GetAttributes();
  if (split_attr.find("axis") != split_attr.end()) {
    auto& axis_attr = split_attr.at("axis");
    axis = utils::HasInt(axis_attr) ? (int)axis_attr.i() : axis;
  } else {
    return false;
  }
  std::cout << "Params part 3" << std::endl;
  auto& quickgelu_attr = quickgelu_node.GetAttributes();
  if (quickgelu_attr.find("alpha") != quickgelu_attr.end()) {
    auto& alpha_attr = quickgelu_attr.at("alpha");
    alpha = utils::HasInt(alpha_attr) ? (int)alpha_attr.i() : alpha;
  } else {
    return false;
  }
  std::cout << "FINAL PARAMS, return true" << std::endl;
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
  std::cout << "FUSE SUBGRAPH Part 1" << std::endl;
  std::string fused_desc =
      "fused " + split_node.Name() + " and " + quickgelu_node.Name() + " and " + mul_node.Name() + " into SplitQuickGelu";

  std::string op_type = "S2SModelSplitQuickGelu";
  std::cout << "FUSE SUBGRAPH Part 2" << std::endl;
  Node& fused_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                   op_type,
                                   fused_desc,
                                   std::array{input},
                                   {},
                                   {},
                                   kMSDomain);

  // add split axis
  // add QuickGelu alpha
  std::cout << "FUSE SUBGRAPH Part 2" << std::endl;
  fused_node.AddAttribute("axis", static_cast<int64_t>(axis));
  fused_node.AddAttribute("alpha", static_cast<int64_t>(alpha));
  std::cout << "FUSE SUBGRAPH Part 3" << std::endl;

  // finalize node fusion (e.g. remove old nodes and shift outputs)
  std::cout << "FUSE SUBGRAPH Part 4" << std::endl;
  fused_node.SetExecutionProviderType(split_node.GetExecutionProviderType());
  std::cout << "FUSE SUBGRAPH Part 5" << std::endl;
  graph_utils::FinalizeNodeFusion(graph, {split_node, quickgelu_node, mul_node}, fused_node);
  std::cout << "FUSE SUBGRAPH Complete" << std::endl;
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
    if (!TrySplitQuickGeluMatch(graph, node, split_node, quickgelu_node, mul_node, logger)) {
    // if (!TrySplitQuickGeluMatch(graph, node, split_node, quickgelu_node, mul_node)) {
      continue;
    }

    NodeArg* input;
    int axis;
    int alpha;
    std::cout << "Get Params Now" << std::endl;
    if (!GetSplitQuickGeluParams(*split_node, *quickgelu_node, input, axis, alpha)) {
      continue;
    }
    std::cout << "FUSE SUBGRAPH NOW" << std::endl;
    FuseSplitQuickGeluSubgraph(graph, *split_node, *quickgelu_node, *mul_node, input, axis, alpha);
    modified = true;

    VLOGF(logger, 1, "Fused S2S Model Split + QuickGelu into S2SModelSplitQuickGelu node.\n");
  }

  return Status::OK();
}

}  // namespace onnxruntime
