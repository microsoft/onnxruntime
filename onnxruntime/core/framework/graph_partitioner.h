// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"
#include "core/framework/fuse_nodes_funcs.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelRegistry;
class KernelRegistryManager;
using TransformLayoutFunction = std::function<Status(Graph& graph, bool& modified, IExecutionProvider& current_ep)>;

class GraphPartitioner {
 public:
  enum class Mode {
    kNormal = 0,
    kAssignOnly = 1,    // assign nodes. no call to Compile. used to create ORT format model support for compiling EPs
    kOrtFormatLoad = 2  // loading ORT format model. Partition with compiling EPs, GraphViewer based Compile.
  };

  //The order of providers represents the user preference.
  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr, const ExecutionProviders& providers)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers) {
  }

  // Run partitioning. Provide compiled_kernel_hashes if mode is kOrtFormatLoad.
  Status Partition(Graph& graph, bool export_dll, FuncManager& func_mgr, 
                   TransformLayoutFunction transform_layout_function,
                   Mode mode = Mode::kNormal,
                   std::unordered_map<std::string, HashValue>* compiled_kernel_hashes = nullptr) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphPartitioner);

#if !defined(ORT_MINIMAL_BUILD)
  Status PartitionOnnxFormatModel(Graph& graph, bool export_dll, FuncManager& func_mgr,
                                  KernelRegistry& fused_kernel_registry, Mode mode,
                                  int& fused_node_unique_id, TransformLayoutFunction transform_layout_function) const;
#endif

  Status PartitionOrtFormatModel(Graph& graph, FuncManager& func_mgr, KernelRegistry& fused_kernel_registry,
                                 std::unordered_map<std::string, HashValue>& compiled_kernel_hashes,
                                 int& fused_node_unique_id, TransformLayoutFunction transform_layout_function) const;

  KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
};
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
