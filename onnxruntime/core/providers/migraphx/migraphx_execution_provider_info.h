// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// Information needed to construct trt execution providers.
struct MIGraphXExecutionProviderInfo {
  std::string target_device;
  int device_id{0};
  bool fp16_enable{false};
  bool int8_enable{false};

  static MIGraphXExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const MIGraphXExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtMIGraphXProviderOptions& info);
};
}  // namespace onnxruntime
