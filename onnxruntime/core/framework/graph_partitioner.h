// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/transform_layout_functions.h"
#include "core/optimizer/graph_optimizer_registry.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelRegistryManager;
class Model;
struct ConfigOptions;
struct EpContextModelGenerationOptions;

class GraphPartitioner {
 public:
  enum class Mode {
    kNormal = 0,
    kAssignOnly = 1,    // assign nodes. no call to Compile. used to create ORT format model support for compiling EPs
    kOrtFormatLoad = 2  // loading ORT format model. Partition with compiling EPs, GraphViewer based Compile.
  };

  // The order of providers represents the user preference.
  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr,
                   const ExecutionProviders& providers,
                   std::unique_ptr<GraphOptimizerRegistry> graph_optimizer_registry)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers),
        graph_optimizer_registry_(std::move(graph_optimizer_registry)) {
  }

  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr,
                   const ExecutionProviders& providers,
                   std::unique_ptr<GraphOptimizerRegistry> graph_optimizer_registry,
                   CheckLoadCancellationFn check_load_cancellation_fn)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers),
        graph_optimizer_registry_(std::move(graph_optimizer_registry)),
        check_load_cancellation_fn_(std::move(check_load_cancellation_fn)) {
  }

  // Run partitioning.
  Status Partition(Graph& graph, FuncManager& func_mgr,
                   const layout_transformation::TransformLayoutFunction& transform_layout_function,
                   const ConfigOptions& config_options,
                   const logging::Logger& logger,
                   Mode mode = Mode::kNormal,
                   const EpContextModelGenerationOptions& ep_context_gen_options = {},
                   const layout_transformation::DebugGraphFn& debug_graph_fn = {}) const;

  bool IsLoadCancellationFlagSet() const {
    return check_load_cancellation_fn_ && check_load_cancellation_fn_();
  }

#ifndef ORT_MINIMAL_BUILD
  /// <summary>
  // Ahead of Time Function inlining. The main purpose of the function is to inline as many
  // functions as possible and delete locally defined functions to reduce the size of the model.
  // This would make other optimizations to be more effective.
  //
  // This function performs GetCapability on the graph and its subgraphs bottom up
  // and inlines any functions that are not claimed by any of the execution providers.
  // This function does not attempt to run layout transformation, and it does not assign EPs.
  // The latter will be done by graph partitioning after Level1 optimizations are done.
  /// </summary>
  /// <param name="model">model instance</param>
  /// <param name="execution_providers">execution providers considered</param>
  /// <param name="kernel_registry_manager">registry manager</param>
  /// <param name="logger">session logger</param>
  /// <returns></returns>
  Status InlineFunctionsAOT(Model& model,
                            const ExecutionProviders& execution_providers,
                            const KernelRegistryManager& kernel_registry_manager,
                            const logging::Logger& logger) const;
#endif

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphPartitioner);

  KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
  std::unique_ptr<GraphOptimizerRegistry> graph_optimizer_registry_;
  CheckLoadCancellationFn check_load_cancellation_fn_;
};

}  // namespace onnxruntime
