// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace onnxruntime {

namespace profiling {

std::string demangle(const char* name);
std::string demangle(const std::string& name);

}  // namespace profiling
}  // namespace onnxruntime
