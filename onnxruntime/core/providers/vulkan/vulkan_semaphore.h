// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

namespace onnxruntime {

class VulkanSemaphore {
 public:
  VulkanSemaphore(const VkDevice& logical_device);
  virtual ~VulkanSemaphore();

  VkSemaphore Get() const {
    return semaphore_;
  }

 private:
  const VkDevice& logical_device_;
  VkSemaphore semaphore_;
};
}  // namespace onnxruntime