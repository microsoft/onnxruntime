// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_silu_fusion.h"

#include <cstdio>

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

bool HasInput(const Node& node, size_t index) {
  return index < node.InputDefs().size() && node.InputDefs()[index] != nullptr && !node.InputDefs()[index]->Name().empty();
}

const Node* GetInputNode(const Graph& graph, const Node& node, size_t input_index) {
  const auto* edge = graph_utils::GetInputEdge(node, static_cast<int>(input_index));
  return edge == nullptr ? nullptr : graph.GetNode(edge->GetNode().Index());
}

bool IsSupportedMul(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "Mul", {7, 13, 14});
}

bool IsSupportedSigmoid(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6, 13});
}

bool IsMatMulNBitsWithoutZeroPointOrGroupIdx(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMulNBits", {1}, kMSDomain) &&
         !HasInput(node, 3) && !HasInput(node, 4);
}

int64_t GetIntAttr(const Node& node, const char* name, int64_t default_value, bool required = false) {
  const auto* attr = graph_utils::GetNodeAttribute(node, name);
  if (attr == nullptr) {
    ORT_ENFORCE(!required, "Missing required attribute ", name, " on node ", node.Name());
    return default_value;
  }

  return attr->i();
}

bool HasSingleNonGraphConsumer(const Graph& graph, const Node& node) {
  return !graph.NodeProducesGraphOutput(node) && optimizer_utils::CheckOutputEdges(graph, node, 1);
}

bool IsFuseCandidate(const Graph& graph,
                     const Node& gate_matmul,
                     const Node& up_matmul,
                     const Node& sigmoid,
                     const Node& silu_mul,
                     const Node& final_mul) {
  if (!IsMatMulNBitsWithoutZeroPointOrGroupIdx(gate_matmul) || !IsMatMulNBitsWithoutZeroPointOrGroupIdx(up_matmul) ||
      !IsSupportedSigmoid(sigmoid) || !IsSupportedMul(silu_mul) || !IsSupportedMul(final_mul)) {
    return false;
  }

  if (!HasSingleNonGraphConsumer(graph, up_matmul) || !HasSingleNonGraphConsumer(graph, sigmoid) ||
      !HasSingleNonGraphConsumer(graph, silu_mul)) {
    return false;
  }

  if (graph.NodeProducesGraphOutput(gate_matmul) || gate_matmul.GetOutputEdgesCount() != 2) {
    return false;
  }

  if (gate_matmul.InputDefs().empty() || up_matmul.InputDefs().empty() ||
      gate_matmul.InputDefs()[0] != up_matmul.InputDefs()[0]) {
    return false;
  }

  if (sigmoid.InputDefs()[0] != gate_matmul.OutputDefs()[0]) {
    return false;
  }

  const bool silu_mul_matches =
      (silu_mul.InputDefs()[0] == gate_matmul.OutputDefs()[0] && silu_mul.InputDefs()[1] == sigmoid.OutputDefs()[0]) ||
      (silu_mul.InputDefs()[1] == gate_matmul.OutputDefs()[0] && silu_mul.InputDefs()[0] == sigmoid.OutputDefs()[0]);
  if (!silu_mul_matches) {
    return false;
  }

  const bool final_mul_matches =
      (final_mul.InputDefs()[0] == silu_mul.OutputDefs()[0] && final_mul.InputDefs()[1] == up_matmul.OutputDefs()[0]) ||
      (final_mul.InputDefs()[1] == silu_mul.OutputDefs()[0] && final_mul.InputDefs()[0] == up_matmul.OutputDefs()[0]);
  if (!final_mul_matches) {
    return false;
  }

  const int64_t gate_k = GetIntAttr(gate_matmul, "K", -1, true);
  const int64_t up_k = GetIntAttr(up_matmul, "K", -1, true);
  const int64_t gate_n = GetIntAttr(gate_matmul, "N", -1, true);
  const int64_t up_n = GetIntAttr(up_matmul, "N", -1, true);
  const int64_t gate_bits = GetIntAttr(gate_matmul, "bits", 4);
  const int64_t up_bits = GetIntAttr(up_matmul, "bits", 4);
  const int64_t gate_block_size = GetIntAttr(gate_matmul, "block_size", -1, true);
  const int64_t up_block_size = GetIntAttr(up_matmul, "block_size", -1, true);
  const int64_t gate_accuracy_level = GetIntAttr(gate_matmul, "accuracy_level", 0);
  const int64_t up_accuracy_level = GetIntAttr(up_matmul, "accuracy_level", 0);

  return gate_k == up_k && gate_n == up_n && gate_bits == up_bits && gate_block_size == up_block_size &&
         gate_accuracy_level == up_accuracy_level;
}

}  // namespace

Status MatMulNBitsSiluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr) {
      continue;
    }

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!IsSupportedMul(node)) {
      continue;
    }

    const auto& node_ep = node.GetExecutionProviderType();
    if (!node_ep.empty() && node_ep != kWebGpuExecutionProvider) {
      continue;
    }

    const Node* input0 = GetInputNode(graph, node, 0);
    const Node* input1 = GetInputNode(graph, node, 1);
    if (input0 == nullptr || input1 == nullptr) {
      continue;
    }

    const Node* silu_mul = nullptr;
    const Node* up_matmul = nullptr;
    if (IsSupportedMul(*input0) && IsMatMulNBitsWithoutZeroPointOrGroupIdx(*input1)) {
      silu_mul = input0;
      up_matmul = input1;
    } else if (IsSupportedMul(*input1) && IsMatMulNBitsWithoutZeroPointOrGroupIdx(*input0)) {
      silu_mul = input1;
      up_matmul = input0;
    } else {
      continue;
    }

    const Node* silu_input0 = GetInputNode(graph, *silu_mul, 0);
    const Node* silu_input1 = GetInputNode(graph, *silu_mul, 1);
    if (silu_input0 == nullptr || silu_input1 == nullptr) {
      continue;
    }

    const Node* gate_matmul = nullptr;
    const Node* sigmoid = nullptr;
    if (IsMatMulNBitsWithoutZeroPointOrGroupIdx(*silu_input0) && IsSupportedSigmoid(*silu_input1)) {
      gate_matmul = silu_input0;
      sigmoid = silu_input1;
    } else if (IsMatMulNBitsWithoutZeroPointOrGroupIdx(*silu_input1) && IsSupportedSigmoid(*silu_input0)) {
      gate_matmul = silu_input1;
      sigmoid = silu_input0;
    } else {
      continue;
    }

    if (!IsFuseCandidate(graph, *gate_matmul, *up_matmul, *sigmoid, *silu_mul, node)) {
      continue;
    }

    LOGS(logger, INFO) << "MatMulNBitsSiluFusion: matched candidate final_mul='" << node.Name()
                       << "' gate='" << gate_matmul->Name() << "' up='" << up_matmul->Name()
                       << "' sigmoid='" << sigmoid->Name() << "' silu_mul='" << silu_mul->Name()
                       << "' attrs={K=" << GetIntAttr(*gate_matmul, "K", -1, true)
                       << ", N=" << GetIntAttr(*gate_matmul, "N", -1, true)
                       << ", bits=" << GetIntAttr(*gate_matmul, "bits", 4)
                       << ", block_size=" << GetIntAttr(*gate_matmul, "block_size", -1, true)
                       << ", accuracy_level=" << GetIntAttr(*gate_matmul, "accuracy_level", 0)
                       << "}";

    LOGS(logger, INFO) << "MatMulNBitsSiluFusion: EP state final_mul='" << node.GetExecutionProviderType()
                       << "' gate='" << gate_matmul->GetExecutionProviderType()
                       << "' up='" << up_matmul->GetExecutionProviderType()
                       << "' sigmoid='" << sigmoid->GetExecutionProviderType()
                       << "' silu_mul='" << silu_mul->GetExecutionProviderType() << "'";

    if ((!gate_matmul->GetExecutionProviderType().empty() && gate_matmul->GetExecutionProviderType() != kWebGpuExecutionProvider) ||
        (!up_matmul->GetExecutionProviderType().empty() && up_matmul->GetExecutionProviderType() != kWebGpuExecutionProvider) ||
        (!sigmoid->GetExecutionProviderType().empty() && sigmoid->GetExecutionProviderType() != kWebGpuExecutionProvider) ||
        (!silu_mul->GetExecutionProviderType().empty() && silu_mul->GetExecutionProviderType() != kWebGpuExecutionProvider)) {
      LOGS(logger, INFO) << "MatMulNBitsSiluFusion: skipping candidate due to non-WebGPU EP assignment.";
      continue;
    }

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", GetIntAttr(*gate_matmul, "K", -1, true)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("N", GetIntAttr(*gate_matmul, "N", -1, true)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", GetIntAttr(*gate_matmul, "bits", 4)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", GetIntAttr(*gate_matmul, "block_size", -1, true)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", GetIntAttr(*gate_matmul, "accuracy_level", 0)), attrs);

    NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);

    InlinedVector<NodeArg*> fused_inputs{
        const_cast<NodeArg*>(gate_matmul->InputDefs()[0]),
        const_cast<NodeArg*>(gate_matmul->InputDefs()[1]),
        const_cast<NodeArg*>(gate_matmul->InputDefs()[2]),
        HasInput(*gate_matmul, 5) ? const_cast<NodeArg*>(gate_matmul->InputDefs()[5]) : &empty_arg,
        const_cast<NodeArg*>(up_matmul->InputDefs()[1]),
        const_cast<NodeArg*>(up_matmul->InputDefs()[2]),
        HasInput(*up_matmul, 5) ? const_cast<NodeArg*>(up_matmul->InputDefs()[5]) : &empty_arg,
    };

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("MatMulNBitsSiluMul"),
                                     "MatMulNBitsSiluMul",
                                     "fused MatMulNBits gate/up projections with SiLU multiply",
                                     fused_inputs,
                                     {},
                                     &attrs,
                                     kMSDomain);
    fused_node.SetExecutionProviderType(kWebGpuExecutionProvider);

    LOGS(logger, INFO) << "MatMulNBitsSiluFusion: created fused node '" << fused_node.Name()
               << "' from final_mul='" << node.Name() << "'";

    graph_utils::FinalizeNodeFusion(graph,
                                    {std::ref(const_cast<Node&>(*gate_matmul)),
                                     std::ref(const_cast<Node&>(*up_matmul)),
                                     std::ref(const_cast<Node&>(*sigmoid)),
                                     std::ref(const_cast<Node&>(*silu_mul)),
                                     std::ref(node)},
                                    fused_node);

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
