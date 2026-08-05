// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <span>
#include <string>

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

std::string CalculateProgramCacheKey(const ProgramBase& program,
                                     std::span<uint32_t> inputs_segments,
                                     std::span<uint32_t> outputs_segments,
                                     bool is_1d_dispatch);

}  // namespace webgpu
}  // namespace onnxruntime
