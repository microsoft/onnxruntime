// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

namespace onnxruntime {
namespace cuda {

// Unified allowlist of ops eligible for NHWC layout conversion in both the
// bundled CUDA EP and the CUDA plugin EP.  Maintaining a single source of truth
// prevents silent divergence between the two implementations.

inline bool IsNhwcEligibleOnnxOp(std::string_view op_type) {
  // Alphabetical order for easy maintenance.
  return op_type == "AveragePool" ||
         op_type == "BatchNormalization" ||
         op_type == "Conv" ||
         op_type == "ConvTranspose" ||
         op_type == "DepthToSpace" ||
         op_type == "GlobalAveragePool" ||
         op_type == "GlobalMaxPool" ||
         op_type == "GridSample" ||
         op_type == "LRN" ||
         op_type == "MaxPool" ||
         op_type == "SpaceToDepth";
}

inline bool IsNhwcEligibleMsOp(std::string_view op_type) {
  return op_type == "GridSample";
}

// Returns true if the given (domain, op_type) pair is eligible for NHWC
// conversion.  |domain| should be kOnnxDomain ("") or kMSDomain
// ("com.microsoft").
inline bool IsNhwcEligible(std::string_view domain, std::string_view op_type) {
  if (domain.empty()) {
    return IsNhwcEligibleOnnxOp(op_type);
  }
  if (domain == "com.microsoft") {
    return IsNhwcEligibleMsOp(op_type);
  }
  return false;
}

}  // namespace cuda
}  // namespace onnxruntime
