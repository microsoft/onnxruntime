// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <cinttypes>

namespace onnxruntime {
namespace webgpu {

extern std::string TypeSnippet(uint32_t component, std::string data_type);
extern std::string BiasSnippet(bool has_bias);

}  // namespace webgpu
}  // namespace onnxruntime
