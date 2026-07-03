// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <string_view>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace ONNX_NAMESPACE {
class FunctionProto;
}

namespace onnxruntime {

/// Adjacency list representation of a local function call graph.
/// Keys and values are string_views into stable storage (e.g. map keys that outlive this structure).
using LocalFunctionCallGraph = InlinedHashMap<std::string_view, InlinedVector<std::string_view>>;

/// Build a call graph adjacency list from model local functions.
/// String views in the returned graph point into the keys of @p model_local_functions.
Status BuildLocalFunctionCallGraph(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    LocalFunctionCallGraph& call_graph);

/// Validate that a call graph contains no cycles.
/// Returns an error with the cycle path if a cycle is detected.
Status ValidateCallGraphAcyclic(const LocalFunctionCallGraph& call_graph);

/// Convenience: build the call graph from model local functions and validate acyclicity.
Status ValidateModelLocalFunctionAcyclic(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions);

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
