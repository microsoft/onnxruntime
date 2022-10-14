// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <variant>

#if !defined(ORT_MINIMAL_BUILD)
#include <functional>

#include "core/common/status.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
using onnxruntime::common::Status;
namespace ONNX_NAMESPACE {
class OpSchema;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

namespace onnxruntime {

// Application context objects for various Selector Action Transformer (SAT) modes of operation:
// - SatDirectApplicationContext: Directly apply transformations to the graph.
//     Run Selectors and Actions.
// - SatRuntimeOptimizationSaveContext: Save runtime optimizations separately in the graph for later replay.
//     Run Selectors and save Actions.
// - SatRuntimeOptimizationLoadContext: Load runtime optimizations from the graph and replay them if applicable.
//     Run Actions.

// Context to directly apply optimizations.
struct SatDirectApplicationContext {
};

// Context to save runtime optimizations for later replay.
struct SatRuntimeOptimizationSaveContext {
#if !defined(ORT_MINIMAL_BUILD)
  std::function<Status(const ONNX_NAMESPACE::OpSchema&)> record_produced_node_op_schema;
#endif  // !defined(ORT_MINIMAL_BUILD)
};

// Context to load runtime optimizations and replay them.
struct SatRuntimeOptimizationLoadContext {
};

// Union of SAT application contexts for various modes of operation.
// The default mode is direct application.
using SatApplyContextVariant = std::variant<SatDirectApplicationContext,
                                            SatRuntimeOptimizationSaveContext,
                                            SatRuntimeOptimizationLoadContext>;

}  // namespace onnxruntime
