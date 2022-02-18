// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/dropout_bitmask_rewrite.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// TODO: Dynamically determine whether BitmaskDropout can be run with the existing EP of
// Dropout, in SatisfyCondition? Is this possible?
constexpr std::initializer_list<const char*> kSupportedExecutionProviders = {kCudaExecutionProvider};

Status DropoutBitmaskRewrite::Apply(Graph& graph, Node& dropout_node, RewriteRuleEffect& modified, const logging::Logger&) const {
  Node& new_bitmask_dropout_node = graph.AddNode(/*name=*/graph.GenerateNodeName(dropout_node.Name() + "_bitmask_rewritten"),
                                                 /*op_type=*/"BitmaskDropout",
                                                 /*description=*/"Written from Dropout node",
                                                 /*input_args=*/dropout_node.MutableInputDefs(),
                                                 /*output_args=*/dropout_node.MutableOutputDefs(),
                                                 /*attributes=*/nullptr,
                                                 /*domain=*/kMSDomain);

  const NodeAttributes& dropout_attributes = dropout_node.GetAttributes();
  if (dropout_attributes.count("seed")) {
    new_bitmask_dropout_node.AddAttribute("seed", dropout_attributes.at("seed").f());
  }

  new_bitmask_dropout_node.SetExecutionProviderType(dropout_node.GetExecutionProviderType());

  // Move all input edges from original Dropout to new BitmaskDropout
  for (auto dropout_input_edge : graph_utils::GraphEdge::GetNodeInputEdges(dropout_node)) {
    ORT_ENFORCE(dropout_input_edge.dst_arg_index < 3);
    graph.AddEdge(/*src_node_index=*/dropout_input_edge.src_node,
                  /*dst_node_index=*/new_bitmask_dropout_node.Index(),
                  /*src_arg_slot=*/dropout_input_edge.src_arg_index,
                  /*dst_arg_slot=*/dropout_input_edge.dst_arg_index);
    graph.RemoveEdge(/*src_node_index=*/dropout_input_edge.src_node,
                     /*dst_node_index=*/dropout_input_edge.dst_node,
                     /*src_arg_slot=*/dropout_input_edge.src_arg_index,
                     /*dst_arg_slot=*/dropout_input_edge.dst_arg_index);
  }

  // Move all output edges from original Dropout to new BitmaskDropout
  for (auto dropout_output_edge : graph_utils::GraphEdge::GetNodeOutputEdges(dropout_node)) {
    ORT_ENFORCE(dropout_output_edge.src_arg_index < 2);
    graph.AddEdge(/*src_node_index=*/new_bitmask_dropout_node.Index(),
                  /*dst_node_index=*/dropout_output_edge.dst_node,
                  /*src_arg_slot=*/dropout_output_edge.src_arg_index,
                  /*dst_arg_slot=*/dropout_output_edge.dst_arg_index);
    graph.RemoveEdge(/*src_node_index=*/dropout_output_edge.src_node,
                     /*dst_node_index=*/dropout_output_edge.dst_node,
                     /*src_arg_slot=*/dropout_output_edge.src_arg_index,
                     /*dst_arg_slot=*/dropout_output_edge.dst_arg_index);
  }

  ORT_ENFORCE(dropout_node.GetInputEdgesCount() == 0);
  ORT_ENFORCE(dropout_node.GetOutputEdgesCount() == 0);
  ORT_ENFORCE(graph.RemoveNode(dropout_node.Index()));

  modified = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool DropoutBitmaskRewrite::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // Perform a series of checks. If any fail, rewrite may not be performed.

  // Original Dropout must have opset == 13, as BitmaskDropout only supports opset version 13.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", {13})) {
    return false;
  }

  // If this node does not have a supported execution provider type (likely because BitmaskDropout
  // has no implementation for the specified EP type), rewrite may not be be performed.
  const std::string node_ep = node.GetExecutionProviderType();
  if (std::find(kSupportedExecutionProviders.begin(), kSupportedExecutionProviders.end(), node_ep) == kSupportedExecutionProviders.end()) {
    return false;
  }

  // If Dropout has 2 outputs, this means that the second output (mask) must not be used. In practice,
  // this means that the following conditions must be met:
  //
  // - output 2 not be used as a graph output.
  // - output 2 must not be used as input to any other nodes.
  if (node.OutputDefs().size() >= 2) {
    const NodeArg* mask_output = node.OutputDefs()[1];

    // If mask output is used as a graph output, rewrite is impossible.
    if (graph.IsOutput(mask_output)) {
      return false;
    }

    // If any nodes consume the mask output (use it as input), rewrite is impossible.
    if (!graph.GetConsumerNodes(mask_output->Name()).empty()) {
      return false;
    }
  }

  // If this rewrite has not been invalidated, rewrite is valid.
  return true;
}  // namespace onnxruntime

}  // namespace onnxruntime
