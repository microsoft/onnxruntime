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
  return node.OpType() == "GroupQueryAttention" && node.Domain() == kMSDomain;
}

// Graph::GetProducerNode() / GetConsumerNodes() and the maps behind them are compiled out of a base
// minimal build (include/onnxruntime/core/graph/graph.h, the
// !ORT_MINIMAL_BUILD || ORT_EXTENDED_MINIMAL_BUILD block), and this translation unit is in the base
// minimal source list so the ORT format path can enforce an explicit BNSH request there. Fall back to
// walking the nodes when the maps are unavailable.
//
// The fallback is linear per lookup rather than a hash probe, so a full boundary scan costs
// O(GQA nodes x graph nodes). It is only reached in a minimal build, and there PartitionOrtFormatModel()
// asks for boundaries only when the application actually set the layout option. A full build has the
// maps and can afford to ask on every load, which is what keeps the unfused-Transpose diagnostic
// working for a converted model loaded without the option.
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

const Node* ProducerOf(const Graph& graph, const std::string& arg_name) {
  return graph.GetProducerNode(arg_name);
}

InlinedVector<const Node*> ConsumersOf(const Graph& graph, const std::string& arg_name) {
  const auto consumers = graph.GetConsumerNodes(arg_name);
  return InlinedVector<const Node*>(consumers.begin(), consumers.end());
}

#else

const Node* ProducerOf(const Graph& graph, const std::string& arg_name) {
  for (const auto& node : graph.Nodes()) {
    for (const auto* def : node.OutputDefs()) {
      if (def != nullptr && def->Exists() && def->Name() == arg_name) {
        return &node;
      }
    }
  }
  return nullptr;
}

InlinedVector<const Node*> ConsumersOf(const Graph& graph, const std::string& arg_name) {
  InlinedVector<const Node*> consumers;
  for (const auto& node : graph.Nodes()) {
    for (const auto* def : node.InputDefs()) {
      if (def != nullptr && def->Exists() && def->Name() == arg_name) {
        consumers.push_back(&node);
        break;
      }
    }
  }
  return consumers;
}

#endif

// A device copy inserted by MemcpyTransformer. Those run inside TransformGraph, before the optimized
// model is serialized, so a model saved from a non-CPU session can have one spliced between a graph
// boundary and the provider-side nodes: graph input -> MemcpyFromHost -> Transpose -> GQA, or
// GQA -> Transpose -> MemcpyToHost -> graph output. The op type is not schema-backed and carries no
// meaningful domain, so match on the name alone.
bool IsDeviceCopy(const Node& node) {
  return node.OpType() == "MemcpyFromHost" || node.OpType() == "MemcpyToHost";
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

bool FindConvertedPresentValueBoundaryAfterCopies(const Graph& graph, const NodeArg* arg,
                                                  int copy_hops, std::string& boundary_name) {
  if (arg == nullptr || copy_hops > kMaxDeviceCopyHops) {
    return false;
  }

  for (const Node* consumer : ConsumersOf(graph, arg->Name())) {
    if (consumer == nullptr || consumer->OutputDefs().empty()) {
      continue;
    }
    if (IsGqaValueLayoutTranspose(*consumer)) {
      const NodeArg* boundary = TraceGqaBoundaryForwardThroughDeviceCopies(graph, consumer->OutputDefs()[0]);
      if (boundary != nullptr) {
        boundary_name = boundary->Name();
        return true;
      }
      continue;
    }
    if (IsDeviceCopy(*consumer) &&
        FindConvertedPresentValueBoundaryAfterCopies(graph, consumer->OutputDefs()[0],
                                                     copy_hops + 1, boundary_name)) {
      return true;
    }
  }

  return false;
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
  for (int hops = 0; arg != nullptr && hops <= kMaxDeviceCopyHops; ++hops) {
    if (graph.IsOutput(arg)) {
      return arg;
    }

    const NodeArg* next = nullptr;
    for (const Node* consumer : ConsumersOf(graph, arg->Name())) {
      if (consumer != nullptr && IsDeviceCopy(*consumer) && !consumer->OutputDefs().empty()) {
        next = consumer->OutputDefs()[0];
        break;
      }
    }
    if (next == nullptr) {
      return nullptr;
    }
    arg = next;
  }
  return nullptr;
}

const Node* FindValueLayoutTransposeAfterGraphInput(const Graph& graph, const std::string& boundary_name) {
  std::string current = boundary_name;
  for (int hops = 0; hops <= kMaxDeviceCopyHops; ++hops) {
    const NodeArg* copy_output = nullptr;
    for (const Node* consumer : ConsumersOf(graph, current)) {
      if (consumer == nullptr) {
        continue;
      }
      if (IsGqaValueLayoutTranspose(*consumer)) {
        return consumer;
      }
      if (IsDeviceCopy(*consumer) && !consumer->OutputDefs().empty()) {
        copy_output = consumer->OutputDefs()[0];
      }
    }

    if (copy_output == nullptr) {
      return nullptr;
    }
    current = copy_output->Name();
  }
  return nullptr;
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

bool FindConvertedPastValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name) {
  boundary_name.clear();
  if (!HasOperand(node.InputDefs(), kPastValueInputIndex)) {
    return false;
  }

  // Declared graph inputs, including overridable initializers. A boundary that was converted offline
  // may well be initializer-backed, and its baked-in data is already BNHS, so the conversion is real
  // and must be recognized. That is the mirror of ClassifyPastValue() refusing to convert an
  // initializer-backed boundary itself: swapping a declared shape cannot transpose baked-in data, but
  // data that arrived BNHS needs no transposing.
  const Node* transpose = TraceBackToValueLayoutTranspose(graph, node.InputDefs()[kPastValueInputIndex]);
  if (transpose == nullptr || transpose->InputDefs().empty()) {
    return false;
  }

  // Not necessarily adjacent to the boundary: trace back through any device copies.
  const NodeArg* boundary = TraceGqaBoundaryBackThroughDeviceCopies(graph, transpose->InputDefs()[0]);
  if (boundary == nullptr) {
    return false;
  }

  boundary_name = boundary->Name();  // the graph input, not the GQA operand
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
  // other internal BNSH readers, and those must not hide the conversion. Device copies may appear on
  // either side of the Transpose when it and GQA are assigned to different providers.
  return FindConvertedPresentValueBoundaryAfterCopies(graph, arg, 0, boundary_name);
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
