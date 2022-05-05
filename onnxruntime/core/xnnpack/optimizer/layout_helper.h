// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <onnx/onnx_pb.h>
#include "core/common/common.h"

namespace onnxruntime {
Status CreateTransposeNode(::ONNX_NAMESPACE::NodeProto& node, const std::string& node_name,
                           const std::string& input_name,
                           const std::string& output_name, const std::vector<int64_t>& perm);
}  // namespace onnxruntime