// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {

struct VitisAIExecutionProviderInfo;

struct VitisAIProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const VitisAIExecutionProviderInfo& info);
};
}  // namespace onnxruntime
