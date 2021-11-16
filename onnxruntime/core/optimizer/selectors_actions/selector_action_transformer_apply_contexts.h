// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <unordered_map>
#include <variant>

#include "core/framework/kernel_registry_manager.h"
#include "core/graph/basic_types.h"

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
  std::reference_wrapper<const KernelRegistryManager> kernel_registry_manager;
};

// Context to load runtime optimizations and replay them.
struct SatRuntimeOptimizationLoadContext {
  // Note: This is effectively an output parameter.
  std::reference_wrapper<std::unordered_map<NodeIndex, HashValue>> actual_node_index_to_kernel_def_hash;
};

// Union of SAT application contexts for various modes of operation.
// The default mode is direct application.
using SatApplyContextVariant = std::variant<SatDirectApplicationContext,
                                            SatRuntimeOptimizationSaveContext,
                                            SatRuntimeOptimizationLoadContext>;

}  // namespace onnxruntime
