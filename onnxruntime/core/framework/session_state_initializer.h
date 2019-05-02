// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <map>

#include "core/common/const_pointer_container.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/path_lib.h"

namespace onnxruntime {
class ExecutionProviders;
class Graph;
class GraphTransformerManager;
class InsertCastTransformer;
class KernelRegistryManager;
class Node;
class NodeArg;
class SessionState;

namespace logging {
class Logger;
}

// Don't use this class before graph partition is done
class SessionStateInitializer {
 public:
  /**
   *
   * \param graph_loc The file path of where the graph was loaded. e.g. /tmp/test_squeezenet/model.onnx
   */
  SessionStateInitializer(const std::basic_string<PATH_CHAR_TYPE>& graph_loc, onnxruntime::Graph& graph,
                          SessionState& session_state, const ExecutionProviders& providers,
                          KernelRegistryManager& kernel_registry_manager);

  // First perform any transformations and create the execution plan
  common::Status CreatePlan(const Node* parent_node,
                            const ConstPointerContainer<std::vector<NodeArg*>>* outer_scope_node_args,
                            bool enable_sequential_execution);

  // initialize tensors, and save. save kernels and input/output node mappings
  // \param implicit_inputs could be NULL
  common::Status InitializeAndSave(const ConstPointerContainer<std::vector<NodeArg*>>* implicit_inputs);

 private:
  const std::basic_string<PATH_CHAR_TYPE>& graph_loc_;
  onnxruntime::Graph& graph_;
  SessionState& session_state_;

  const ExecutionProviders& execution_providers_;
  KernelRegistryManager& kernel_registry_manager_;
  const logging::Logger& logger_;
};
}  // namespace onnxruntime
