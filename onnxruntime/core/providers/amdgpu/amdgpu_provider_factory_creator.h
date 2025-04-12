// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtAMDGPUProviderOptions;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct AMDGPUProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtAMDGPUProviderOptions* options);
};
}  // namespace onnxruntime
