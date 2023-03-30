// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// data types for execution provider options which only be used internally

// This struct is same as OrtCustomOpDomain defined in inference_session.h,
// but we don't want to include it since it contains many session related definitions 
// which may cuase compile error if we do it.
struct OrtProviderCustomOpDomain {
  std::string domain_;
  std::vector<const OrtCustomOp*> custom_ops_;
};

}  // namespace onnxruntime
