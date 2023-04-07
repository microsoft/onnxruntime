// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct VitisAIProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
      const char* backend_type, int device_id, const char* export_runtime_module,
      const char* load_runtime_module);
};
}  // namespace onnxruntime
