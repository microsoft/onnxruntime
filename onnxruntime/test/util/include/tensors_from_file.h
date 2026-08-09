// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace onnxruntime {
namespace test {

// Read a dictionary of name to float tensors mapping from a text file.
void LoadTensorsFromFile(const std::string& path, std::unordered_map<std::string, std::vector<float>>& tensors);

}  // namespace test
}  // namespace onnxruntime
