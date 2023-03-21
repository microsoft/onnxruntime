// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/framework/fuse_nodes_funcs.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelRegistryManager;

namespace layout_transformer {
// TransformLayoutFunction used by GraphPartitioner when transforming a graph from NCHW to NHWC if the EP has a
// preferred layout of NHWC.
//
// DebugGraphFn can be used to debug the graph modifications made during layout transformation.
// See kDebugLayoutTransformation in /include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h for
// more details.
using DebugGraphFn = std::function<void(const Graph&)>;
using TransformLayoutFunction = std::function<Status(Graph& graph, bool& modified, IExecutionProvider& current_ep,
                                                     const DebugGraphFn& debug_graph_fn)>;
}  // namespace layout_transformer

class GraphPartitioner {
 public:
  enum class Mode {
    kNormal = 0,
    kAssignOnly = 1,    // assign nodes. no call to Compile. used to create ORT format model support for compiling EPs
    kOrtFormatLoad = 2  // loading ORT format model. Partition with compiling EPs, GraphViewer based Compile.
  };

  // The order of providers represents the user preference.
  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr, const ExecutionProviders& providers)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers) {
  }

  // Run partitioning.
  Status Partition(Graph& graph, FuncManager& func_mgr,
                   layout_transformer::TransformLayoutFunction transform_layout_function,
                   Mode mode = Mode::kNormal,
                   const layout_transformer::DebugGraphFn& debug_graph_fn = {}) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphPartitioner);

  KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
};

}  // namespace onnxruntime
