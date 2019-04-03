// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_UTIL_H
#define ONNXRUNTIME_HOSTING_UTIL_H

#include <google/protobuf/stubs/status.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace hosting {

// Generate proper protobuf status from ONNX Runtime status
google::protobuf::util::Status GenerateProtoBufStatus(const onnxruntime::common::Status& onnx_status, const std::string& message);

}  // namespace hosting
}  // namespace onnxruntime

#endif  //ONNXRUNTIME_HOSTING_UTIL_H
