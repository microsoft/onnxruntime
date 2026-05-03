// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_mlp_fusion.h"

#include <algorithm>

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

constexpr const char* kActivationAttrName = "activation";
// The transformer name is generic for future expansion, but the current fused
// pattern and emitted op only support gate activation = "silu".
constexpr const char* kSupportedActivation = "silu";

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

bool IsSupportedSimplifiedLayerNormalization(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "SimplifiedLayerNormalization", {1});
}

bool IsSupportedSkipSimplifiedLayerNormalization(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "SkipSimplifiedLayerNormalization", {1}, kMSDomain);
}

bool IsSupportedMlpNormAnchor(const Node& node) {
  return IsSupportedSimplifiedLayerNormalization(node) || IsSupportedSkipSimplifiedLayerNormalization(node);
}

bool HasProducedOutput(const Node& node, size_t index) {
  return index < node.OutputDefs().size() && node.OutputDefs()[index] != nullptr && !node.OutputDefs()[index]->Name().empty();
}

bool ProducesOnlyOptionalSkipOutputAsGraphOutput(const Graph& graph, const Node& node) {
  const auto graph_outputs = graph.GetNodeOutputsInGraphOutputs(node);
  return std::all_of(graph_outputs.begin(), graph_outputs.end(), [](int output_idx) { return output_idx == 3; });
}

size_t ExpectedNormConsumerEdgeCount(const Node& node) {
  return 2u + ((IsSupportedSkipSimplifiedLayerNormalization(node) && HasProducedOutput(node, 3)) ? 1u : 0u);
}

bool HasExpectedNormConsumers(const Graph& graph, const Node& node) {
  const auto graph_outputs = graph.GetNodeOutputsInGraphOutputs(node);
  const size_t expected_output_edges = ExpectedNormConsumerEdgeCount(node) - graph_outputs.size();
  if (node.GetOutputEdgesCount() != expected_output_edges) {
    return false;
  }

  for (auto output_edge_it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); output_edge_it != end; ++output_edge_it) {
    const auto& output_node = output_edge_it->GetNode();
    const auto output_node_input_arg_idx = static_cast<size_t>(output_edge_it->GetDstArgIndex());
    const bool is_implicit_input_to_output_node = output_node_input_arg_idx >= output_node.InputDefs().size();
    if (is_implicit_input_to_output_node) {
      return false;
    }
  }

  return true;
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

float GetFloatAttr(const Node& node, const char* name, float default_value) {
  const auto* attr = graph_utils::GetNodeAttribute(node, name);
  return attr == nullptr ? default_value : attr->f();
}

bool HasSingleNonGraphConsumer(const Graph& graph, const Node& node) {
  return !graph.NodeProducesGraphOutput(node) && optimizer_utils::CheckOutputEdges(graph, node, 1);
}

const Node* GetNormProducer(const Graph& graph,
                            const Node& gate_matmul,
                            const Node& up_matmul) {
  if (gate_matmul.InputDefs().empty() || up_matmul.InputDefs().empty() ||
      gate_matmul.InputDefs()[0] != up_matmul.InputDefs()[0]) {
    return nullptr;
  }

  const Node* gate_input = GetInputNode(graph, gate_matmul, 0);
  const Node* up_input = GetInputNode(graph, up_matmul, 0);
  if (gate_input == nullptr || gate_input != up_input || !IsSupportedMlpNormAnchor(*gate_input)) {
    return nullptr;
  }

  if (!HasProducedOutput(*gate_input, 0)) {
    return nullptr;
  }

  if (graph.NodeProducesGraphOutput(*gate_input) && !ProducesOnlyOptionalSkipOutputAsGraphOutput(graph, *gate_input)) {
    return nullptr;
  }

  if (!HasExpectedNormConsumers(graph, *gate_input)) {
    return nullptr;
  }

  const size_t min_norm_inputs = IsSupportedSkipSimplifiedLayerNormalization(*gate_input) ? 3u : 2u;
  if (gate_input->InputDefs().size() < min_norm_inputs) {
    return nullptr;
  }

  return gate_input;
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

  return gate_k == up_k && gate_n == up_n &&
         gate_bits == up_bits && gate_bits == 4 &&
         gate_block_size == up_block_size && gate_block_size == 32 &&
         gate_accuracy_level == up_accuracy_level;
}

}  // namespace

Status MatMulNBitsMlpFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
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

    LOGS(logger, VERBOSE) << "MatMulNBitsMlpFusion: matched candidate output_mul='" << node.Name()
                          << "' gate='" << gate_matmul->Name() << "' up='" << up_matmul->Name()
                          << "' sigmoid='" << sigmoid->Name() << "' activation_mul='" << silu_mul->Name()
                          << "' attrs={K=" << GetIntAttr(*gate_matmul, "K", -1, true)
                          << ", N=" << GetIntAttr(*gate_matmul, "N", -1, true)
                          << ", bits=" << GetIntAttr(*gate_matmul, "bits", 4)
                          << ", block_size=" << GetIntAttr(*gate_matmul, "block_size", -1, true)
                          << ", accuracy_level=" << GetIntAttr(*gate_matmul, "accuracy_level", 0)
                          << "}";

    if ((!gate_matmul->GetExecutionProviderType().empty() && gate_matmul->GetExecutionProviderType() != kWebGpuExecutionProvider) ||
        (!up_matmul->GetExecutionProviderType().empty() && up_matmul->GetExecutionProviderType() != kWebGpuExecutionProvider) ||
        (!sigmoid->GetExecutionProviderType().empty() && sigmoid->GetExecutionProviderType() != kWebGpuExecutionProvider) ||
        (!silu_mul->GetExecutionProviderType().empty() && silu_mul->GetExecutionProviderType() != kWebGpuExecutionProvider)) {
      LOGS(logger, VERBOSE) << "MatMulNBitsMlpFusion: skipping candidate due to non-WebGPU EP assignment.";
      continue;
    }

    const Node* norm = GetNormProducer(graph, *gate_matmul, *up_matmul);
    if (norm == nullptr) {
      continue;
    }

    if (!norm->GetExecutionProviderType().empty() && norm->GetExecutionProviderType() != kWebGpuExecutionProvider) {
      continue;
    }

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", GetIntAttr(*gate_matmul, "K", -1, true)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("N", GetIntAttr(*gate_matmul, "N", -1, true)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", GetIntAttr(*gate_matmul, "bits", 4)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", GetIntAttr(*gate_matmul, "block_size", -1, true)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", GetIntAttr(*gate_matmul, "accuracy_level", 0)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute(kActivationAttrName, std::string{kSupportedActivation}), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("epsilon", GetFloatAttr(*norm, "epsilon", 1e-5f)), attrs);

    NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
    const bool is_skip_sln = norm != nullptr && IsSupportedSkipSimplifiedLayerNormalization(*norm);

    InlinedVector<NodeArg*> fused_inputs{
        const_cast<NodeArg*>(norm->InputDefs()[0]),
        is_skip_sln ? const_cast<NodeArg*>(norm->InputDefs()[1]) : &empty_arg,
        const_cast<NodeArg*>(norm->InputDefs()[is_skip_sln ? 2 : 1]),
        const_cast<NodeArg*>(gate_matmul->InputDefs()[1]),
        const_cast<NodeArg*>(gate_matmul->InputDefs()[2]),
        HasInput(*gate_matmul, 5) ? const_cast<NodeArg*>(gate_matmul->InputDefs()[5]) : &empty_arg,
        const_cast<NodeArg*>(up_matmul->InputDefs()[1]),
        const_cast<NodeArg*>(up_matmul->InputDefs()[2]),
        HasInput(*up_matmul, 5) ? const_cast<NodeArg*>(up_matmul->InputDefs()[5]) : &empty_arg,
    };

    InlinedVector<NodeArg*> fused_outputs{const_cast<NodeArg*>(node.OutputDefs()[0])};
    const bool preserve_skip_output = is_skip_sln && norm != nullptr && HasProducedOutput(*norm, 3);
    if (preserve_skip_output) {
      fused_outputs.push_back(const_cast<NodeArg*>(norm->OutputDefs()[3]));
    }

    const auto norm_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*norm);
    const auto gate_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*gate_matmul);
    const auto up_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*up_matmul);
    const auto final_mul_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node);
    const auto norm_output_edges = preserve_skip_output ? graph_utils::GraphEdge::GetNodeOutputEdges(*norm)
                                                        : std::vector<graph_utils::GraphEdge>{};

    const std::string output_mul_name = node.Name();

    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*norm));
    graph.RemoveNode(norm->Index());
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*gate_matmul));
    graph.RemoveNode(gate_matmul->Index());
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*up_matmul));
    graph.RemoveNode(up_matmul->Index());
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*sigmoid));
    graph.RemoveNode(sigmoid->Index());
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*silu_mul));
    graph.RemoveNode(silu_mul->Index());
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("MatMulNBitsMlp"),
                                     "MatMulNBitsMlp",
                                     "fused MatMulNBits gated MLP projections",
                                     fused_inputs,
                                     fused_outputs,
                                     &attrs,
                                     kMSDomain);
    fused_node.SetExecutionProviderType(kWebGpuExecutionProvider);

    LOGS(logger, VERBOSE) << "MatMulNBitsMlpFusion: created fused node '" << fused_node.Name()
                          << "' from output_mul='" << output_mul_name << "'";

    for (const auto& input_edge : norm_input_edges) {
      int fused_input_index = input_edge.dst_arg_index;
      if (!is_skip_sln && input_edge.dst_arg_index == 1) {
        fused_input_index = 2;
      }

      graph.AddEdge(input_edge.src_node, fused_node.Index(), input_edge.src_arg_index, fused_input_index);
    }

    auto add_input_edge_if_present = [&](const std::vector<graph_utils::GraphEdge>& edges,
                                         int source_input_index,
                                         int fused_input_index) {
      for (const auto& input_edge : edges) {
        if (input_edge.dst_arg_index == source_input_index) {
          graph.AddEdge(input_edge.src_node, fused_node.Index(), input_edge.src_arg_index, fused_input_index);
        }
      }
    };

    add_input_edge_if_present(gate_input_edges, 1, 3);
    add_input_edge_if_present(gate_input_edges, 2, 4);
    add_input_edge_if_present(gate_input_edges, 5, 5);
    add_input_edge_if_present(up_input_edges, 1, 6);
    add_input_edge_if_present(up_input_edges, 2, 7);
    add_input_edge_if_present(up_input_edges, 5, 8);

    for (const auto& output_edge : final_mul_output_edges) {
      graph.AddEdge(fused_node.Index(), output_edge.dst_node, 0, output_edge.dst_arg_index);
    }
    if (preserve_skip_output) {
      for (const auto& output_edge : norm_output_edges) {
        if (output_edge.src_arg_index == 3) {
          graph.AddEdge(fused_node.Index(), output_edge.dst_node, 1, output_edge.dst_arg_index);
        }
      }
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
