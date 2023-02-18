// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// data types for execution provider options

using ProviderOptions = std::unordered_map<std::string, std::string>;
using ProviderOptionsVector = std::vector<ProviderOptions>;
using ProviderOptionsMap = std::unordered_map<std::string, ProviderOptions>;

// This struct is same as OrtCustomOpDomain defined in inference_session.h,
// but we shouldn't include it since it contains many session related defines 
// which may cuase compile error if we do it.
struct OrtProviderCustomOpDomain {
  std::string domain_;
  std::vector<const OrtCustomOp*> custom_ops_;
};

}  // namespace onnxruntime
