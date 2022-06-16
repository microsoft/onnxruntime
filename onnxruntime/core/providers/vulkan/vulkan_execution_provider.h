// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vulkan/vulkan.h>

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {
class VulkanExecutionProvider : public IExecutionProvider {
  VulkanExecutionProvider()
      : IExecutionProvider{onnxruntime::kVulkanExecutionProvider} {
  }
};

}  // namespace onnxruntime