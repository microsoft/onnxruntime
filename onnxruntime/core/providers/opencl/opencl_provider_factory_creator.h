// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/framework/provider_options.h"


namespace onnxruntime {
struct SessionOptions;
// defined in provider_bridge_ort.cc
struct OpenCLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create();
};
}  // namespace onnxruntime
