// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace graph_utils {

inline std::string RecomputeName(const std::string& name) {
  return name + "_recompute";
}

}  // namespace graph_utils
}  // namespace onnxruntime
