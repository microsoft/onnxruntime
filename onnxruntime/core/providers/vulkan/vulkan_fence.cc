// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_fence.h"

namespace onnxruntime {

VulkanFence::VulkanFence(const VkDevice& logical_device) : logical_device_(logical_device) {
#ifdef VK_USE_PLATFORM_WIN32_KHR
  // which one is correct on windows ?
  VkExportFenceCreateInfoKHR efci;
  // VkExportFenceWin32HandleInfoKHR efci;
  efci.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
  efci.pNext = NULL;
  efci.sType = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
  VkExportFenceCreateInfoKHR efci;
  efci.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
  efci.pNext = NULL;
#if VK_USE_PLATFORM_ANDROID_KHR  // current android only support VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
  efci.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
#else
  efci.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
#endif

  VkFenceCreateInfo fence_create_info{
      /* .sType = */ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      /* .pNext = */ nullptr,
      /* .flags = */ 0,
  };

  VK_CALL(vkCreateFence(logical_device_, &fence_create_info, nullptr, &fence_));
}

VulkanFence::~VulkanFence() {
  VK_CALL_RETURNS_VOID(vkDestroyFence(logical_device_, fence_, nullptr));
}

VkResult VulkanFence::RawWait() const {
  auto status = VK_TIMEOUT;

  do {
    status = vkWaitForFences(logical_device_, 1, &fence_, VK_TRUE, 5000000000);
  } while (status == VK_TIMEOUT);

  return status;
}

VkResult VulkanFence::Wait() const {
  return RawWait();
}

VkResult VulkanFence::Reset() const {
  return vkResetFences(logical_device_, 1, &fence_);
}

}  // namespace onnxruntime
