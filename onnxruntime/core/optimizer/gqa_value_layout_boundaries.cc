// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_value_layout_boundaries.h"

#include <array>
#include <cstdint>
#include <vector>

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
  return node.OpType().compare("GroupQueryAttention") == 0 && node.Domain().compare(kMSDomain) == 0;
}

const Node* ProducerOf(const Graph& graph, const std::string& arg_name) {
  return graph.GetProducerNode(arg_name);
}

template <typename Result, typename Visitor>
const Result* FindConsumer(const Graph& graph, const std::string& arg_name, Visitor&& visit) {
  for (const Node* consumer : graph.GetConsumerNodes(arg_name)) {
    if (const Result* result = visit(consumer)) {
      return result;
    }
  }
  return nullptr;
}

// A device copy inserted by MemcpyTransformer. Those run inside TransformGraph, before the optimized
// model is serialized, so a model saved from a non-CPU session can have one spliced between a graph
// boundary and the provider-side nodes: graph input -> MemcpyFromHost -> Transpose -> GQA, or
// GQA -> Transpose -> MemcpyToHost -> graph output. The op type is not schema-backed and carries no
// meaningful domain, so match on the name alone.
bool IsDeviceCopy(const Node& node) {
  return node.OpType().compare("MemcpyFromHost") == 0 || node.OpType().compare("MemcpyToHost") == 0;
}

// MemcpyTransformer inserts at most one copy per boundary, but walk a few hops so a future pass that
// chains them still resolves, while staying bounded against a malformed graph.
constexpr int kMaxDeviceCopyHops = 4;

const Node* TraceBackToValueLayoutTranspose(const Graph& graph, const NodeArg* arg) {
  for (int hops = 0; arg != nullptr && hops <= kMaxDeviceCopyHops; ++hops) {
    const Node* producer = ProducerOf(graph, arg->Name());
    if (producer == nullptr) {
      return nullptr;
    }
    if (IsGqaValueLayoutTranspose(*producer)) {
      return producer;
    }
    if (!IsDeviceCopy(*producer) || producer->InputDefs().empty()) {
      return nullptr;
    }
    arg = producer->InputDefs()[0];
  }
  return nullptr;
}

const NodeArg* TraceBoundaryForwardThroughDeviceCopies(const Graph& graph, const NodeArg* arg, int copy_hops,
                                                       bool needs_transpose = false) {
  if (arg == nullptr || copy_hops > kMaxDeviceCopyHops) {
    return nullptr;
  }
  if (!needs_transpose && graph.IsOutput(arg)) {
    return arg;
  }
  return FindConsumer<NodeArg>(graph, arg->Name(), [&](const Node* consumer) -> const NodeArg* {
    if (consumer == nullptr || consumer->OutputDefs().empty()) {
      return nullptr;
    }
    if (needs_transpose && IsGqaValueLayoutTranspose(*consumer)) {
      return TraceBoundaryForwardThroughDeviceCopies(graph, consumer->OutputDefs()[0], 0);
    }
    if (IsDeviceCopy(*consumer)) {
      return TraceBoundaryForwardThroughDeviceCopies(graph, consumer->OutputDefs()[0], copy_hops + 1, needs_transpose);
    }
    return nullptr;
  });
}

}  // namespace

bool IsGqaValueLayoutTranspose(const Node& node) {
  if (node.OpType().compare("Transpose") != 0 || node.Domain().compare(kOnnxDomain) != 0) {
    return false;
  }

  for (const auto& [name, attribute] : node.GetAttributes()) {
    if (name.compare("perm") == 0) {
      if (static_cast<size_t>(attribute.ints_size()) != kValueLayoutPerm.size()) {
        return false;
      }
      for (size_t index = 0; index < kValueLayoutPerm.size(); ++index) {
        if (attribute.ints(static_cast<int>(index)) != kValueLayoutPerm[index]) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

namespace {
bool ContainsByName(const std::vector<const NodeArg*>& args, const NodeArg* arg) {
  for (const auto* candidate : args) {
    if (candidate != nullptr && candidate->Name() == arg->Name()) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool IsGqaDeclaredGraphInput(const Graph& graph, const NodeArg* arg) {
  return arg != nullptr && ContainsByName(graph.GetInputsIncludingInitializers(), arg);
}

bool IsGqaNonInitializerGraphInput(const Graph& graph, const NodeArg* arg) {
  return arg != nullptr && ContainsByName(graph.GetInputs(), arg);
}

const NodeArg* TraceGqaBoundaryBackThroughDeviceCopies(const Graph& graph, const NodeArg* arg) {
  for (int hops = 0; arg != nullptr && hops <= kMaxDeviceCopyHops; ++hops) {
    if (IsGqaDeclaredGraphInput(graph, arg)) {
      return arg;
    }

    const Node* producer = ProducerOf(graph, arg->Name());
    if (producer == nullptr || !IsDeviceCopy(*producer) || producer->InputDefs().empty()) {
      return nullptr;
    }
    arg = producer->InputDefs()[0];
  }
  return nullptr;
}

const NodeArg* TraceGqaBoundaryForwardThroughDeviceCopies(const Graph& graph, const NodeArg* arg) {
  return TraceBoundaryForwardThroughDeviceCopies(graph, arg, 0);
}

namespace {
const Node* FindValueLayoutTransposeAfterCopies(const Graph& graph, const std::string& arg_name, int copy_hops) {
  if (copy_hops > kMaxDeviceCopyHops) {
    return nullptr;
  }
  return FindConsumer<Node>(graph, arg_name, [&](const Node* consumer) -> const Node* {
    if (consumer == nullptr) {
      return nullptr;
    }
    if (IsGqaValueLayoutTranspose(*consumer)) {
      return consumer;
    }
    if (IsDeviceCopy(*consumer) && !consumer->OutputDefs().empty()) {
      return FindValueLayoutTransposeAfterCopies(graph, consumer->OutputDefs()[0]->Name(), copy_hops + 1);
    }
    return nullptr;
  });
}
}  // namespace

const Node* FindValueLayoutTransposeAfterGraphInput(const Graph& graph, const std::string& boundary_name) {
  return FindValueLayoutTransposeAfterCopies(graph, boundary_name, 0);
}

const Node* FindValueLayoutTransposeBeforeGraphOutput(const Graph& graph, const std::string& boundary_name) {
  std::string current = boundary_name;
  for (int hops = 0; hops <= kMaxDeviceCopyHops; ++hops) {
    const Node* producer = ProducerOf(graph, current);
    if (producer == nullptr) {
      return nullptr;
    }
    if (IsGqaValueLayoutTranspose(*producer)) {
      return producer;
    }
    if (!IsDeviceCopy(*producer) || producer->InputDefs().empty()) {
      return nullptr;
    }
    current = producer->InputDefs()[0]->Name();
  }
  return nullptr;
}

namespace {
const NodeArg* ConvertedPastValueBoundary(const Graph& graph, const Node& node) {
  if (!HasOperand(node.InputDefs(), kPastValueInputIndex)) {
    return nullptr;
  }

  // Declared graph inputs, including overridable initializers. A boundary that was converted offline
  // may well be initializer-backed, and its baked-in data is already BNHS, so the conversion is real
  // and must be recognized. That is the mirror of ClassifyPastValue() refusing to convert an
  // initializer-backed boundary itself: swapping a declared shape cannot transpose baked-in data, but
  // data that arrived BNHS needs no transposing.
  const Node* transpose = TraceBackToValueLayoutTranspose(graph, node.InputDefs()[kPastValueInputIndex]);
  if (transpose == nullptr || transpose->InputDefs().empty()) {
    return nullptr;
  }

  // Not necessarily adjacent to the boundary: trace back through any device copies.
  return TraceGqaBoundaryBackThroughDeviceCopies(graph, transpose->InputDefs()[0]);
}

const NodeArg* ConvertedPresentValueBoundary(const Graph& graph, const Node& node) {
  if (!HasOperand(node.OutputDefs(), kPresentValueOutputIndex)) {
    return nullptr;
  }

  const NodeArg* arg = node.OutputDefs()[kPresentValueOutputIndex];

  // An operand that is itself a graph output is an application-visible BNSH boundary in its own
  // right, not the internal intermediate of a converted node, even if something downstream also
  // transposes it to a second graph output.
  if (graph.IsOutput(arg)) {
    return nullptr;
  }

  // Search the consumers rather than requiring a single one: the BNSH result may legitimately feed
  // other internal BNSH readers, and those must not hide the conversion. Device copies may appear on
  // either side of the Transpose when it and GQA are assigned to different providers.
  return TraceBoundaryForwardThroughDeviceCopies(graph, arg, 0, true);
}
}  // namespace

bool FindConvertedPastValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name) {
  boundary_name.clear();
  const NodeArg* boundary = ConvertedPastValueBoundary(graph, node);
  if (boundary != nullptr) {
    boundary_name = boundary->Name();
  }
  return boundary != nullptr;
}

bool FindConvertedPresentValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name) {
  boundary_name.clear();
  const NodeArg* boundary = ConvertedPresentValueBoundary(graph, node);
  if (boundary != nullptr) {
    boundary_name = boundary->Name();
  }
  return boundary != nullptr;
}

namespace {
// Counts GQA nodes at any depth below `graph`, not including `graph` itself.
size_t CountGqaNodesInSubgraphs(const Graph& graph) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) {
    for (const Graph* subgraph : node.GetSubgraphs()) {
      if (subgraph == nullptr) {
        continue;
      }
      for (const auto& subgraph_node : subgraph->Nodes()) {
        if (IsGroupQueryAttention(subgraph_node)) {
          ++count;
        }
      }
      count += CountGqaNodesInSubgraphs(*subgraph);
    }
  }
  return count;
}
}  // namespace

GqaNodeCounts CountGqaNodes(const Graph& graph) {
  GqaNodeCounts counts;
  for (const auto& node : graph.Nodes()) {
    if (IsGroupQueryAttention(node)) {
      ++counts.in_main_graph;
    }
  }
  counts.in_subgraphs = CountGqaNodesInSubgraphs(graph);
  return counts;
}

bool HasConvertedGqaValueLayoutBoundaries(const Graph& graph) {
  for (int index = 0; index < graph.MaxNodeIndex(); ++index) {
    const Node* node = graph.GetNode(static_cast<NodeIndex>(index));
    if (node == nullptr || !IsGroupQueryAttention(*node)) {
      continue;
    }

    if (ConvertedPastValueBoundary(graph, *node) != nullptr ||
        ConvertedPresentValueBoundary(graph, *node) != nullptr) {
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
