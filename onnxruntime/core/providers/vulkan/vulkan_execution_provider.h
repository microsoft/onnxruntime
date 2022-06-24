// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {
class VulkanExecutionProvider : public IExecutionProvider {
  VulkanExecutionProvider();
  ~VulkanExecutionProvider();

 private:
  VkInstance vulkan_instance;
  VkPhysicalDevice vulkan_physical_device;
};

}  // namespace onnxruntime