// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

std::string CalculateProgramCacheKey(const ProgramBase& program, bool is_1d_dispatch);

}  // namespace webgpu
}  // namespace onnxruntime
