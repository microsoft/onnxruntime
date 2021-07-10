// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "helper.h"

namespace onnxruntime {
namespace nnapi {

struct OpSupportCheckParams {
  OpSupportCheckParams(int32_t android_feature_level, bool use_nchw)
      : android_feature_level(android_feature_level),
        use_nchw(use_nchw) {
  }

  int32_t android_feature_level = 0;
  bool use_nchw = false;
};

class IOpSupportChecker {
 public:
  virtual ~IOpSupportChecker() = default;

  // Check if an operator is supported
  virtual bool IsOpSupported(const InitializedTensorSet& initializers, const Node& node, const OpSupportCheckParams& params) const = 0;
};

// Get the lookup table with IOpSupportChecker delegates for different onnx operators
// Note, the lookup table should have same number of entries as the result of CreateOpBuilders()
// in op_builder.h
const std::unordered_map<std::string, const IOpSupportChecker*>& GetOpSupportCheckers();

}  // namespace nnapi
}  // namespace onnxruntime