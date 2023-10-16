// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs/headers.
#include <memory>

// 1st-party libs/headers.
#include "core/providers/providers.h"
#include "core/framework/provider_options.h"


namespace onnxruntime {

struct AMDUnifiedExecutionProviderInfo;

struct AMDUnifiedProviderFactoryCreator {
  static std::shared<IExecutionProviderFactory> Create(
      const ProviderOptions& provider_options);
};

}  // namespace onnxruntime
