// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
// Don't check if the op is Deprecated. In ONNX Runtime's world, there is no deprecation.
bool IsSupportedOptypeVersionAndDomain(const Node& node, const std::string& op_type,
                                       std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                       std::string_view domain) {
  return (node.OpType() == op_type && graph_utils::MatchesOpSinceVersion(node, versions) &&
          graph_utils::MatchesOpSetDomain(node, domain));
}

// If the op has multiple versions, here we require it must have a single implementation that can work across all the
// versions. Because in the fusion, we discarded the op version information.
bool IsFusableActivation(const Node& node) {
  return IsSupportedOptypeVersionAndDomain(node, "Elu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "HardSigmoid", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "LeakyRelu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13, 14}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Selu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6, 13}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Softplus", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Softsign", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Tanh", {6, 13}, kOnnxDomain) ||
#ifndef DISABLE_CONTRIB_OPS
         IsSupportedOptypeVersionAndDomain(node, "ScaledTanh", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "ParametricSoftplus", {1}, kOnnxDomain) ||
#endif
         IsSupportedOptypeVersionAndDomain(node, "ThresholdedRelu", {1, 10}, kOnnxDomain);
}
}  // namespace

Status GemmActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {7, 9, 11, 13}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) || node.GetOutputEdgesCount() != 1) {
      continue;
    }

    NodeArg* node_output = node.MutableOutputDefs()[0];
    auto data_type = node_output->TypeAsProto()->tensor_type().elem_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      // FusedGemm is only registered for float data type in fused_gemm.cc!
      continue;
    }

    const Node& next_node = *(node.OutputNodesBegin());
    if (!IsFusableActivation(next_node) || next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    if (graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    Node& gemm_node = node;
    Node& act_node = *graph.GetNode(next_node.Index());  // get mutable reference

    Node& fused_gemm = graph.AddNode(graph.GenerateNodeName("fused " + gemm_node.Name()), "FusedGemm",
                                     "fused Gemm " + gemm_node.Name() + "with activation " + act_node.OpType(),
                                     gemm_node.MutableInputDefs(), {}, &gemm_node.GetAttributes(), kMSDomain);

    // Add a new attribute to specify the activation type
    fused_gemm.AddAttribute("activation", act_node.OpType());

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_gemm.SetExecutionProviderType(gemm_node.GetExecutionProviderType());

    // Add optional attributes for activations
    const NodeAttributes& attrs = act_node.GetAttributes();
    for (const auto& attr : attrs) {
      AttributeProto fused_gemm_attr(attr.second);
      fused_gemm_attr.set_name("activation_" + attr.first);
      fused_gemm.AddAttributeProto(std::move(fused_gemm_attr));
    }

    // move output definitions and edges from act_node to fused_gemm. delete gemm_node and act_node.
    graph_utils::FinalizeNodeFusion(graph, {gemm_node, act_node}, fused_gemm);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
