// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/framework/provider_options.h"

struct OrtMIGraphXProviderOptions;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct MIGraphXProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtMIGraphXProviderOptions* options);
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions&);
};
}  // namespace onnxruntime
