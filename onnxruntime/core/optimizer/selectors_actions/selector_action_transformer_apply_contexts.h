// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <variant>

#include "core/framework/kernel_registry_manager.h"

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
};

// Union of SAT application contexts for various modes of operation.
// The default mode is direct application.
using SatApplyContextVariant = std::variant<SatDirectApplicationContext,
                                            SatRuntimeOptimizationSaveContext,
                                            SatRuntimeOptimizationLoadContext>;

}  // namespace onnxruntime
