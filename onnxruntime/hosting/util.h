// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_UTIL_H
#define ONNXRUNTIME_HOSTING_UTIL_H

#include "core/common/status.h"

#include <google/protobuf/stubs/status.h>

namespace onnxruntime {
namespace hosting {

google::protobuf::util::Status GenerateProtoBufStatus(onnxruntime::common::Status onnx_status, const std::string& message);

}  // namespace hosting
}  // namespace onnxruntime

#endif  //ONNXRUNTIME_HOSTING_UTIL_H
