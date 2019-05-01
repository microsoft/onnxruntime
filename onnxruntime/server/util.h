// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <google/protobuf/stubs/status.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace server {

// Generate protobuf status from ONNX Runtime status
google::protobuf::util::Status GenerateProtobufStatus(const onnxruntime::common::Status& onnx_status, const std::string& message);

}  // namespace server
}  // namespace onnxruntime

