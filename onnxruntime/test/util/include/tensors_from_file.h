// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace onnxruntime {
namespace test {

// Read float tensors from a file
void load_tensors_from_file(const std::string& path, std::unordered_map<std::string, std::vector<float>>& tensors);

}  // namespace test
}  // namespace onnxruntime
