// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct CoreMLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(uint32_t coreml_flags);
};
}  // namespace onnxruntime
