// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "helper.h"

namespace onnxruntime {
namespace nnapi {

struct OPSupportCheckParams {
  int32_t android_sdk_ver = 0;
  bool use_nchw = false;
};

class IOpSupportChecker {
 public:
  virtual ~IOpSupportChecker() = default;

  // Check if an operator is supported
  virtual bool IsOpSupported(const std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto&>& initializers,
                             const Node& node, const OPSupportCheckParams& params) const = 0;
};

// Generate a lookup table with IOpSupportChecker delegates for different onnx operators
// Note, the lookup table should have same number of entries as the result of CreateOpBuilders()
// in op_builder.h
std::unordered_map<std::string, std::shared_ptr<IOpSupportChecker>> CreateOpSupportCheckers();

}  // namespace nnapi
}  // namespace onnxruntime