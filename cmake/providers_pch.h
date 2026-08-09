// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Core framework headers (highest compilation time impact)
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

// Graph-related headers
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"

// ONNX schema definitions
#include "onnx/defs/schema.h"

// Windows-specific headers (if applicable)
#ifdef _WIN32
#include <windows.h>
#endif
