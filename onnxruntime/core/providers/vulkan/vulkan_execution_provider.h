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

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanInstance);

 private:
  VkInstance vulkan_instance_;
};

class VulkanExecutionProvider : public IExecutionProvider {
  VulkanExecutionProvider();
  ~VulkanExecutionProvider();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanExecutionProvider);

 private:
  std::shared_ptr<VulkanInstance> vulkan_instance_;
  VkPhysicalDevice vulkan_physical_device_;
  uint32_t vulkan_queue_family_index_;
  VkDevice vulkan_logical_device_;
  VkPhysicalDeviceProperties vulkan_device_properties_;
  VkQueue vulkan_queue_;
  VkPhysicalDeviceMemoryProperties vulkan_device_memory_properties_;
};

}  // namespace onnxruntime