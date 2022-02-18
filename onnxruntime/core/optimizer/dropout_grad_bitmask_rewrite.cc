// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/dropout_grad_bitmask_rewrite.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

namespace {

// TODO: Unit tests for this rewrite pattern. Some example test .onnx files exist in "bitmask_dropout_grad_gen.py"
// for this rewrite pattern, but none are implemented yet in "graph_transform_tests.cc".
//
// Look under "GraphTransformationTests.BitmaskDropout*" for examples of similiar tests for BitmaskDropoutRewrite.

/**
 * @brief Fully replaces a node with a newly constructed node, deleting the original node from the graph.
 * 
 * Input/output defs and attributes are copied from the old node to the new node. Input/output edges
 * are also fully moved from the old node to the new node. The new node type must have input/ouput defs
 * that match up fully with the old node. The new node will also have the same EP type as the old node.
 * 
 * After this function returns, the old node will be deleted.
 * 
 * @param graph The graph in which to perform this replacement.
 * @param node The old node, which will be replaced and removed.
 * @param name The name of the new node.
 * @param op_type The op type of the new node.
 * @param description A description for the new node.
 * @param domain The domain to create the new node in.
 * @return Node& The newly created node.
 */
Node& FullyReplaceNode(Graph& graph,
                       Node& node,
                       const std::string& name,
                       const std::string& op_type,
                       const std::string& description,
                       const std::string& domain) {
  Node& new_node = graph.AddNode(/*name=*/name,
                                 /*op_type=*/op_type,
                                 /*description=*/description,
                                 /*input_args=*/node.MutableInputDefs(),
                                 /*output_args=*/node.MutableOutputDefs(),
                                 /*attributes=*/&node.GetAttributes(),
                                 /*domain=*/domain);

  // Assign the new node to have the same EP type.
  new_node.SetExecutionProviderType(node.GetExecutionProviderType());

  // Move all input edges from old node to new node.
  for (auto input_edge : graph_utils::GraphEdge::GetNodeInputEdges(node)) {
    graph.AddEdge(/*src_node_index=*/input_edge.src_node,
                  /*dst_node_index=*/new_node.Index(),
                  /*src_arg_slot=*/input_edge.src_arg_index,
                  /*dst_arg_slot=*/input_edge.dst_arg_index);
    graph.RemoveEdge(/*src_node_index=*/input_edge.src_node,
                     /*dst_node_index=*/input_edge.dst_node,
                     /*src_arg_slot=*/input_edge.src_arg_index,
                     /*dst_arg_slot=*/input_edge.dst_arg_index);
  }

  // Move all output edges from original node to new node.
  for (auto output_edge : graph_utils::GraphEdge::GetNodeOutputEdges(node)) {
    graph.AddEdge(/*src_node_index=*/new_node.Index(),
                  /*dst_node_index=*/output_edge.dst_node,
                  /*src_arg_slot=*/output_edge.src_arg_index,
                  /*dst_arg_slot=*/output_edge.dst_arg_index);
    graph.RemoveEdge(/*src_node_index=*/output_edge.src_node,
                     /*dst_node_index=*/output_edge.dst_node,
                     /*src_arg_slot=*/output_edge.src_arg_index,
                     /*dst_arg_slot=*/output_edge.dst_arg_index);
  }

  ORT_ENFORCE(node.GetInputEdgesCount() == 0);
  ORT_ENFORCE(node.GetOutputEdgesCount() == 0);
  ORT_ENFORCE(graph.RemoveNode(node.Index()));

  return new_node;
}

}  // namespace

// TODO: Dynamically determine whether BitmaskDropoutGrad can be run with the existing EP of
// DropoutGrad, in SatisfyCondition? Is this possible?
constexpr std::initializer_list<const char*> kSupportedExecutionProviders = {kCudaExecutionProvider};

Status DropoutGradBitmaskRewrite::Apply(Graph& graph, Node& dropout_node, RewriteRuleEffect& modified, const logging::Logger& logger) const {
  std::cout << "dropout grad replace happening!\n";
  Node& new_bitmask_dropout_node = FullyReplaceNode(/*graph=*/graph,
                                                    /*node=*/dropout_node,
                                                    /*name=*/graph.GenerateNodeName(dropout_node.Name() + "_bitmask_rewritten"),
                                                    /*op_type=*/"BitmaskDropout",
                                                    /*description=*/"Written from Dropout node",
                                                    /*domain=*/kMSDomain);

  if (new_bitmask_dropout_node.OutputDefs().size() >= 2) {
    NodeArg* mask_output = new_bitmask_dropout_node.MutableOutputDefs()[1];
    
    // Update mask output def to be uint32, instead of bool.
    //
    // TODO: Ensure this def has correct output size/dims. Should be (num_elements + 31) / 32.
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(TensorProto::UINT32);
    ORT_THROW_IF_ERROR(mask_output->UpdateTypeAndShape(type_proto, true, true, logger));

    for (Node* consumer : graph.GetMutableConsumerNodes(mask_output->Name())) {
      Node& dropout_grad_node = *consumer;
      ORT_ENFORCE(dropout_grad_node.OpType() == "DropoutGrad");
      Node& new_bitmask_dropout_grad_node = FullyReplaceNode(/*graph=*/graph,
                                                             /*node=*/dropout_grad_node,
                                                             /*name=*/graph.GenerateNodeName(dropout_grad_node.Name() + "_bitmask_rewritten"),
                                                             /*op_type=*/"BitmaskDropoutGrad",
                                                             /*description=*/"Written from DropoutGrad node",
                                                             /*domain=*/kMSDomain);

      ORT_ENFORCE(new_bitmask_dropout_node.GetExecutionProviderType() ==
                  new_bitmask_dropout_grad_node.GetExecutionProviderType());
    }
  }

  modified = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

bool DropoutGradBitmaskRewrite::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // Perform a series of checks. If any fail, rewrite may not be performed.

  std::cout << "starting dropout grad rewrite check\n";
  // Original Dropout must have opset 12 or 13, as BitmaskDropoutGrad only supports Dropout opset versions 12/13.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", {12, 13})) {
    std::cout << "unsupported dropout version\n";
    return false;
  }

  // If this node does not have a supported execution provider type (likely because BitmaskDropoutGrad
  // has no implementation for the specified EP type), rewrite may not be be performed.
  const std::string node_ep = node.GetExecutionProviderType();
  if (std::find(kSupportedExecutionProviders.begin(), kSupportedExecutionProviders.end(), node_ep) == kSupportedExecutionProviders.end()) {
    std::cout << "unsupported EP: " << node_ep << "\n";
    return false;
  }

  // If DropoutGrad has 2 outputs, this means that the second output (mask) must not be used. In practice,
  // this means that the following conditions must be met:
  //
  // - output 2 not be used as a graph output.
  // - output 2 must not be used as input to any other nodes.
  if (node.OutputDefs().size() >= 2) {
    const NodeArg* mask_output = node.OutputDefs()[1];

    // If mask output is used as a graph output, rewrite is impossible.
    if (graph.IsOutput(mask_output)) {
      std::cout << "unsupported node output\n";
      return false;
    }

    for (const Node* consumer : graph.GetConsumerNodes(mask_output->Name())) {
      // All consumers of mask output must be DropoutGrad with opset version number 1, as this is the only supported
      // version of BitmaskDropoutGrad.
      //
      // If any other node types consume the mask output, rewrite is not possible, as these other nodes know nothing
      // about the new format.
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(*consumer, "DropoutGrad", {1}, kMSDomain)) {
        std::cout << "unsupported consumer version name: " << consumer->Name() << ", type: " << consumer->OpType() << "\n";
        return false;
      }

      // Consumer node must have a supported EP, and it must be the same as the Dropout node's EP.
      const std::string consumer_ep = node.GetExecutionProviderType();
      if (consumer_ep != node_ep || std::find(kSupportedExecutionProviders.begin(), kSupportedExecutionProviders.end(), consumer_ep) == kSupportedExecutionProviders.end()) {
        std::cout << "unsupported consumer ep : " << consumer_ep << ", name: " << consumer->Name() << ", type: " << consumer->OpType() << "\n";
        return false;
      }
    }
  }

  std::cout << "dropout grad replacement passed checks!\n";
  // If this rewrite has not been invalidated, rewrite is valid.
  return true;
}  // namespace onnxruntime

}  // namespace onnxruntime
