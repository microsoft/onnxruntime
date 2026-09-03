// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_value_layout_boundaries.h"

#include <array>
#include <cstdint>

#include "core/graph/constants.h"

namespace onnxruntime {

namespace {

// GroupQueryAttention operand positions. See docs/ContribOperators.md#com.microsoft.GroupQueryAttention.
constexpr size_t kPastValueInputIndex = 4;
constexpr size_t kPresentValueOutputIndex = 2;

// Swaps the last two dimensions of a rank-4 tensor.
constexpr std::array<int64_t, 4> kValueLayoutPerm{0, 1, 3, 2};

bool HasOperand(const ConstPointerContainer<std::vector<NodeArg*>>& defs, size_t index) {
  return index < defs.size() && defs[index] != nullptr && defs[index]->Exists();
}

bool IsGroupQueryAttention(const Node& node) {
  return node.OpType() == "GroupQueryAttention" && node.Domain() == kMSDomain;
}

}  // namespace

bool IsGqaValueLayoutTranspose(const Node& node) {
  if (node.OpType() != "Transpose" || node.Domain() != kOnnxDomain) {
    return false;
  }

  // Read the attribute directly rather than through graph_utils, so this stays usable from the
  // minimal build without pulling the optimizer helpers in with it.
  const auto& attributes = node.GetAttributes();
  const auto perm = attributes.find("perm");
  if (perm == attributes.end() || static_cast<size_t>(perm->second.ints_size()) != kValueLayoutPerm.size()) {
    return false;
  }

  for (size_t i = 0; i < kValueLayoutPerm.size(); ++i) {
    if (perm->second.ints(static_cast<int>(i)) != kValueLayoutPerm[i]) {
      return false;
    }
  }

  return true;
}

bool IsGqaApplicationInput(const Graph& graph, const NodeArg* arg) {
  if (arg == nullptr) {
    return false;
  }

  for (const auto* graph_input : graph.GetInputs()) {
    if (graph_input != nullptr && graph_input->Name() == arg->Name()) {
      return true;
    }
  }

  return false;
}

bool FindConvertedPastValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name) {
  boundary_name.clear();
  if (!HasOperand(node.InputDefs(), kPastValueInputIndex)) {
    return false;
  }

  const Node* producer = graph.GetProducerNode(node.InputDefs()[kPastValueInputIndex]->Name());
  if (producer == nullptr || !IsGqaValueLayoutTranspose(*producer) ||
      !IsGqaApplicationInput(graph, producer->InputDefs()[0])) {
    return false;
  }

  boundary_name = producer->InputDefs()[0]->Name();  // the graph input, not the GQA operand
  return true;
}

bool FindConvertedPresentValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name) {
  boundary_name.clear();
  if (!HasOperand(node.OutputDefs(), kPresentValueOutputIndex)) {
    return false;
  }

  const NodeArg* arg = node.OutputDefs()[kPresentValueOutputIndex];

  // An operand that is itself a graph output is an application-visible BNSH boundary in its own
  // right, not the internal intermediate of a converted node, even if something downstream also
  // transposes it to a second graph output.
  if (graph.IsOutput(arg)) {
    return false;
  }

  // Search the consumers rather than requiring a single one: the BNSH result may legitimately feed
  // other internal BNSH readers, and those must not hide the conversion.
  for (const Node* consumer : graph.GetConsumerNodes(arg->Name())) {
    if (consumer != nullptr && IsGqaValueLayoutTranspose(*consumer) && graph.IsOutput(consumer->OutputDefs()[0])) {
      boundary_name = consumer->OutputDefs()[0]->Name();  // the graph output, not the GQA operand
      return true;
    }
  }

  return false;
}

GqaValueLayoutBoundaries FindConvertedGqaValueLayoutBoundaries(const Graph& graph) {
  GqaValueLayoutBoundaries boundaries;

  for (const auto& node : graph.Nodes()) {
    if (!IsGroupQueryAttention(node)) {
      continue;
    }

    std::string boundary_name;
    if (FindConvertedPastValueBoundary(graph, node, boundary_name)) {
      boundaries.past_value_inputs.push_back(boundary_name);
    }
    if (FindConvertedPresentValueBoundary(graph, node, boundary_name)) {
      boundaries.present_value_outputs.push_back(boundary_name);
    }
  }

  return boundaries;
}

}  // namespace onnxruntime
