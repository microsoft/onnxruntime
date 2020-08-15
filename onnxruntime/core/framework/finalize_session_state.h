// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <map>

#include "core/common/const_pointer_container.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_allocator.h"
#include "core/framework/session_options.h"
#include "core/platform/path_lib.h"

namespace onnxruntime {
class KernelRegistryManager;
class Node;
class SessionState;

Status FinalizeSessionState(SessionState& session_state,
                            const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                            KernelRegistryManager& kernel_registry_manager,
                            _In_opt_ const Node* parent_node,
                            const SessionOptions& session_options);

}  // namespace onnxruntime
