// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

class VulkanInstance {
 public:
  VulkanInstance();
  ~VulkanInstance();
  VkInstance Get() const;

 private:
  VkInstance vulkan_instance;
};

class VulkanExecutionProvider : public IExecutionProvider {
  VulkanExecutionProvider();
  ~VulkanExecutionProvider();

 private:
  std::shared_ptr<VulkanInstance> vulkan_instance;
  VkPhysicalDevice vulkan_physical_device;
  uint32_t vulkan_queue_family_index;
  VkDevice vulkan_logical_device;
  VkPhysicalDeviceProperties vulkan_device_properties;
  VkQueue vulkan_queue;
  VkPhysicalDeviceMemoryProperties vulkan_device_memory_properties;
};

}  // namespace onnxruntime