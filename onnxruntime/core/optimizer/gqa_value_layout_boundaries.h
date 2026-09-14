// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/inlined_containers.h"
#include "core/graph/graph.h"

namespace onnxruntime {

// Accepted values of the kOrtSessionOptionsGqaValueLayout session option.
constexpr const char* kGqaValueLayoutBNSH = "BNSH";
constexpr const char* kGqaValueLayoutBNHS = "BNHS";

/**
The application-visible boundaries whose com.microsoft.GroupQueryAttention Value cache is BNHS.

Either because GqaValueLayoutTransformer converted them in this session, or because the model already
arrived that way. Graph input and output names are stable across partitioning, which is what makes
them usable as an anchor for the post-partition diagnostic.
*/
struct GqaValueLayoutBoundaries {
  InlinedVector<std::string> past_value_inputs;      // graph inputs declaring BNHS
  InlinedVector<std::string> present_value_outputs;  // graph outputs declaring BNHS

  bool Empty() const { return past_value_inputs.empty() && present_value_outputs.empty(); }
};

// Is this a Transpose node that swaps the last two dimensions of a rank-4 tensor, i.e. BNSH <-> BNHS?
bool IsGqaValueLayoutTranspose(const Node& node);

// Is `arg` declared as a graph input, initializer-backed or not? An overridable initializer counts:
// the application may bind over it, so it is a boundary it can observe. Use this to recognize a
// boundary that already carries the conversion.
bool IsGqaDeclaredGraphInput(const Graph& graph, const NodeArg* arg);

// Is `arg` a graph input the application must supply? Excludes initializers, which carry baked-in
// data. Use this to decide whether an unconverted boundary may be converted: the declared shape can
// be swapped, but an initializer's data cannot, so an initializer-backed one is rejected instead.
bool IsGqaNonInitializerGraphInput(const Graph& graph, const NodeArg* arg);

// Walks back / forward from `arg` through any device copy nodes (MemcpyFromHost / MemcpyToHost) to
// the graph input or graph output it connects to, or nullptr if it does not reach one. Returns `arg`
// itself when it is already the boundary.
//
// MemcpyTransformer runs inside TransformGraph, before an optimized model is serialized, so a model
// saved from a non-CPU session can have a copy spliced between a boundary and the provider-side
// nodes. Exposed so the transformer can tell a genuinely internal cache apart from an
// application-visible one that merely sits behind a copy.
const NodeArg* TraceGqaBoundaryBackThroughDeviceCopies(const Graph& graph, const NodeArg* arg);
const NodeArg* TraceGqaBoundaryForwardThroughDeviceCopies(const Graph& graph, const NodeArg* arg);

// From an application boundary, walks past any device copies and returns the value-layout Transpose on
// the other side, or nullptr if there is none. The inverse direction of the Trace* helpers above, for
// the post-partition diagnostic: it starts from a recorded boundary name and asks whether the
// Transpose is still there, which the same MemcpyFromHost / MemcpyToHost nodes would otherwise hide.
const Node* FindValueLayoutTransposeAfterGraphInput(const Graph& graph, const std::string& boundary_name);
const Node* FindValueLayoutTransposeBeforeGraphOutput(const Graph& graph, const std::string& boundary_name);

// If this node's past_value already arrives through a value-layout Transpose from a graph input,
// possibly with device copies on either side of the Transpose, returns true and sets boundary_name
// to that graph input.
bool FindConvertedPastValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name);

// If this node's present_value already leaves through a value-layout Transpose to a graph output,
// possibly with device copies on either side of the Transpose, returns true and sets boundary_name
// to that graph output.
bool FindConvertedPresentValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name);

/**
Where a graph's com.microsoft.GroupQueryAttention nodes sit relative to the main graph.

Used to explain why a BNHS request converted nothing. From the main graph alone, a model with no GQA
at all and one whose GQA lives inside a Loop body or BeamSearch decoder look identical -- both simply
have nothing to convert -- but only the second leaves the application binding BNHS buffers to a
boundary that is still BNSH, so the two deserve different messages.
*/
struct GqaNodeCounts {
  size_t in_main_graph = 0;
  size_t in_subgraphs = 0;  // at any depth

  bool Any() const { return in_main_graph != 0 || in_subgraphs != 0; }
};

GqaNodeCounts CountGqaNodes(const Graph& graph);

/**
Finds every application boundary of a graph that already carries the BNHS conversion.

Shared by the transformer and the ORT format load path to enforce an explicit BNSH request and drive
the unfused-Transpose diagnostic. Compiled only when ORT_ENABLE_GQA_VALUE_LAYOUT is defined.
*/
GqaValueLayoutBoundaries FindConvertedGqaValueLayoutBoundaries(const Graph& graph);

// Uses the same boundary rules without collecting names.
bool HasConvertedGqaValueLayoutBoundaries(const Graph& graph);

}  // namespace onnxruntime
