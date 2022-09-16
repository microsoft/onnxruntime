// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

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

}  // namespace onnxruntime