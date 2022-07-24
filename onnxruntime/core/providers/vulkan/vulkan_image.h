// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_memory_pool.h"
#include "core/framework/data_types.h"

namespace onnxruntime {

class VulkanImage {
 public:
  VulkanImage(VulkanMemoryPool& pool, bool seperate, const std::vector<int64_t>& dims,
              MLDataType data_type);

  VulkanImage(VulkanMemoryPool& pool, bool seperate, int w, int h, MLDataType data_type)
      : VulkanImage(vulkan_memory_pool_, seperate, std::vector<int64_t>{w, h}, data_type) {
  }

  virtual ~VulkanImage();

  inline int GetWidth() const {
    return std::get<1>(vulkan_image_info_);
  }

  inline int GetHeight() const {
    return std::get<2>(vulkan_image_info_);
  }

  inline int GetDepth() const {
    return std::get<3>(vulkan_image_info_);
  }

  inline std::vector<int64_t> GetDims() const {
    return vulkan_image_dims_;
  }

  inline VkImage GetImage() const {
    return vulkan_image_and_view_.first;
  }

  inline VkImageView GetView() const {
    return vulkan_image_and_view_.second;
  }

  inline VkFormat GetImageFormat() const {
    return std::get<4>(vulkan_image_info_);
  }

  void Release();

  void ResetBarrier() {
    vulkan_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  }

  VkImageLayout GetLayout() const {
    return vulkan_image_layout_;
  }

  void BarrierWrite(VkCommandBuffer buffer) const;

  void BarrierRead(VkCommandBuffer buffer) const;

 private:
  std::tuple<VkImageType, uint32_t, uint32_t, uint32_t, VkFormat> vulkan_image_info_;
  std::pair<VkImage, VkImageView> vulkan_image_and_view_;
  const VkDevice& vulkan_logical_device_;
  std::vector<int64_t> vulkan_image_dims_;
  VulkanMemoryPool& vulkan_memory_pool_;
  std::pair<void*, int> vulkan_image_memory_;
  mutable VkImageLayout vulkan_image_layout_;
  mutable VkAccessFlagBits vulkan_image_access_flags_;
};

}  // namespace onnxruntime