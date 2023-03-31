// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// This struct is same as OrtCustomOpDomain defined in inference_session.h and only be used by EP internally.
// We don't want to include inference_session.h in EP since it contains many session related definitions which may cause compile error.
struct OrtProviderCustomOpDomain {
  std::string domain_;
  std::vector<const OrtCustomOp*> custom_ops_;
};

}  // namespace onnxruntime
