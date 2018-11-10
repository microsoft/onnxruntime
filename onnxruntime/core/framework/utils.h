// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
class Node;
class Graph;
}  // namespace onnxruntime

namespace onnxruntime {
class ExecutionProviders;
class KernelDef;
class KernelRegistryManager;
class SessionState;

namespace logging {
class Logger;
}

namespace utils {
const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const onnxruntime::Node& node);

const KernelDef* GetKernelDef(const onnxruntime::Graph& graph,
                              const KernelRegistryManager& kernel_registry,
                              const onnxruntime::NodeIndex node_id);

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const ONNXRuntimeAllocatorInfo& allocator_info);

AllocatorPtr GetAllocator(const SessionState& session_state,
                          const ONNXRuntimeAllocatorInfo& allocator_info);
}  // namespace utils
}  // namespace onnxruntime
