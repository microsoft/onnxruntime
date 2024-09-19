// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "vaip/dll_safe.h"
#include "vaip/my_ort.h"
namespace vaip {

using namespace onnxruntime;
using NodeInput = vaip_core::NodeInput;
///
vaip_core::DllSafe<std::vector<NodeInput>> node_get_inputs(const Node& node);

/// to support multiple outputs
vaip_core::DllSafe<std::vector<const NodeArg*>> node_get_output_node_args(const Node& node);
}  // namespace vaip
