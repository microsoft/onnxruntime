// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/framework/kernel_registry_manager.h"

namespace onnxruntime {

struct RuntimeOptimizationSaveContext {
  std::reference_wrapper<const KernelRegistryManager> kernel_registry_manager;
};

}  // namespace onnxruntime
