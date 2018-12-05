// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_types.h"
#include "core/graph/graph_viewer.h"
#include "onnx/defs/data_type_utils.h"

namespace onnxruntime {
namespace utils {
MLDataType GetMLDataType(const onnxruntime::NodeArg& arg);
}  // namespace utils
}  // namespace onnxruntime
