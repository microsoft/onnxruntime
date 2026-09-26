// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

SplitKConfig CreateSplitKConfig(std::string_view architecture);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
