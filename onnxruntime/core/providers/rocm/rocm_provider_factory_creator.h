// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtROCMProviderOptions;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct RocmProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtROCMProviderOptions* provider_options);
};
}  // namespace onnxruntime
