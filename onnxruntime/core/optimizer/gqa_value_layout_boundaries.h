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

// If this node's past_value already arrives through a value-layout Transpose from a graph input,
// returns true and sets boundary_name to that graph input.
bool FindConvertedPastValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name);

// If this node's present_value already leaves through a value-layout Transpose to a graph output,
// returns true and sets boundary_name to that graph output.
bool FindConvertedPresentValueBoundary(const Graph& graph, const Node& node, std::string& boundary_name);

/**
Finds every application boundary of a graph that already carries the BNHS conversion.

Lives in its own translation unit, compiled in every build flavour including minimal, because the ORT
format load path needs it to enforce an explicit BNSH request and to drive the unfused-Transpose
diagnostic. The rest of the transformer is full-build only.
*/
GqaValueLayoutBoundaries FindConvertedGqaValueLayoutBoundaries(const Graph& graph);

}  // namespace onnxruntime
