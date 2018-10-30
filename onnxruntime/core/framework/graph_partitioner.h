// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelRegistryManager;
class SessionState;

class GraphPartitioner {
 public:
  //The order of providers represents the user preference.
  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr,
                   const ExecutionProviders& providers)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers) {}

  Status Partition(onnxruntime::Graph& graph, const SessionState& session_state) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphPartitioner);

  KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
};
}  // namespace onnxruntime
