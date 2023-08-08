// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

namespace onnxruntime {

// Mapping from params signature to kernel id
using KernelMap = std::unordered_map<std::string, int>;

struct TuningResults {
  std::string ep;

  // Validates if these results are compatible with the libraries, the validation process is EP defined
  std::unordered_map<std::string, std::string> validators;

  // Mapping from Op signature to Op's tuning result
  std::unordered_map<std::string, KernelMap> results;
};

}  // namespace onnxruntime
