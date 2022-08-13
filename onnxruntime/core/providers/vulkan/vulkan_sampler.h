// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

namespace onnxruntime {

class VulkanSampler {
 public:
  VulkanSampler(const VkDevice& logical_device, VkFilter filter = VK_FILTER_NEAREST,
                VkSamplerAddressMode mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);

  virtual ~VulkanSampler();

  VkSampler Get() const {
    return sampler_;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanSampler);

 private:
  VkSampler sampler_;
  const VkDevice& logical_device_;
};

}  // namespace onnxruntime