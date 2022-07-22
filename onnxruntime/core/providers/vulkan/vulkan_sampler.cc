// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_sampler.h"

namespace onnxruntime {

VulkanSampler::VulkanSampler(const VkDevice& vulkan_logical_device, VkFilter filter, VkSamplerAddressMode mode) : vulkan_logical_device_(vulkan_logical_device) {
  VkSamplerCreateInfo samplerInfo;
  ::memset(&samplerInfo, 0, sizeof(samplerInfo));

  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = filter;
  samplerInfo.minFilter = filter;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = mode;
  samplerInfo.addressModeV = mode;
  samplerInfo.addressModeW = mode;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 0.0f;

  VK_CALL(vkCreateSampler(vulkan_logical_device_, &samplerInfo, nullptr, &sampler_));
}

VulkanSampler::~VulkanSampler() {
  VK_CALL_RETURNS_VOID(vkDestroySampler(vulkan_logical_device_, sampler_, nullptr));
}

}  // namespace onnxruntime
