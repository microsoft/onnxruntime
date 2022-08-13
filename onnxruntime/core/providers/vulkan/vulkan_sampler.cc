// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_sampler.h"

namespace onnxruntime {

VulkanSampler::VulkanSampler(const VkDevice& logical_device, VkFilter filter, VkSamplerAddressMode mode) : logical_device_(logical_device) {
  VkSamplerCreateInfo sampler_info;
  ::memset(&sampler_info, 0, sizeof(sampler_info));

  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = filter;
  sampler_info.minFilter = filter;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler_info.addressModeU = mode;
  sampler_info.addressModeV = mode;
  sampler_info.addressModeW = mode;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.maxAnisotropy = 1.0f;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = 0.0f;

  VK_CALL(vkCreateSampler(logical_device_, &sampler_info, nullptr, &sampler_));
}

VulkanSampler::~VulkanSampler() {
  VK_CALL_RETURNS_VOID(vkDestroySampler(logical_device_, sampler_, nullptr));
}

}  // namespace onnxruntime
