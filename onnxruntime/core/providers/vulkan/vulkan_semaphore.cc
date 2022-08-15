// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_semaphore.h"

namespace onnxruntime {

VulkanSemaphore::VulkanSemaphore(const VkDevice& logical_device) : logical_device_(logical_device) {
  VkSemaphoreCreateInfo semaphore_create_info = {};

  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.flags = 0;
  semaphore_create_info.pNext = nullptr;

  VK_CALL(vkCreateSemaphore(logical_device_, &semaphore_create_info, nullptr, &semaphore_));
}

VulkanSemaphore::~VulkanSemaphore() {
  vkDestroySemaphore(logical_device_, semaphore_, nullptr);
}

}  // namespace onnxruntime
