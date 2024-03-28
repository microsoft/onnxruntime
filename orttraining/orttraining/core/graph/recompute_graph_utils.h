// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace graph_utils {

constexpr const char* kRecomputeFlag = "_recompute";

inline std::string RecomputeName(const std::string& name) {
  return name + kRecomputeFlag;
}

}  // namespace graph_utils
}  // namespace onnxruntime
