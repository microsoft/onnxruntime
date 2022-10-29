// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct DMLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id, bool skip_software_device_check);
};
}  // namespace onnxruntime
