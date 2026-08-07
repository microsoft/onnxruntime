// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime {
namespace webgpu {
namespace detail {

// Format a WGSL shader compilation info structure into a human-readable multi-line diagnostic
// string. Each message is annotated with the offending source line and a caret pointing at the
// column reported by the compiler. Exposed for testing.
std::string FormatShaderCompilationInfo(std::string_view code, const wgpu::CompilationInfo* info);

// Prepend 1-based line numbers to each line of `code`. Used to make the annotated WGSL source
// easy to correlate with compiler-reported line numbers. Exposed for testing.
std::string AnnotateShaderWithLineNumbers(std::string_view code);

}  // namespace detail
}  // namespace webgpu
}  // namespace onnxruntime
