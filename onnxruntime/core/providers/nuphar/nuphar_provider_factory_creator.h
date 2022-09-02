// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct NupharProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(bool allow_unaligned_buffers, const char* settings);
};
}  // namespace onnxruntime
