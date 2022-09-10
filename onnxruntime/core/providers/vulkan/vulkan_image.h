// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_memory_allocation_helper.h"
#include "core/framework/data_types.h"

namespace onnxruntime {

class VulkanImage {
 public:
  VulkanImage(VulkanMemoryAllocationHelper& memory_alloc_helper, const std::vector<int64_t>& dims,
              MLDataType data_type);

  VulkanImage(VulkanMemoryAllocationHelper& memory_alloc_helper, int64_t w, int64_t h, MLDataType data_type)
      : VulkanImage(memory_alloc_helper, std::vector<int64_t>{w, h}, data_type) {
  }

  virtual ~VulkanImage();

  inline int64_t GetWidth() const {
    return std::get<1>(image_info_);
  }

  inline int64_t GetHeight() const {
    return std::get<2>(image_info_);
  }

  inline int64_t GetDepth() const {
    return std::get<3>(image_info_);
  }

  inline std::vector<int64_t> GetDims() const {
    return image_dims_;
  }

  inline VkImage GetImage() const {
    return image_and_view_.first;
  }

  inline VkImageView GetView() const {
    return image_and_view_.second;
  }

  inline VkFormat GetImageFormat() const {
    return std::get<4>(image_info_);
  }

  void Release();

  void ResetBarrier() {
    image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  }

  VkImageLayout GetLayout() const {
    return image_layout_;
  }

  void BarrierWrite(VkCommandBuffer buffer);

  void BarrierRead(VkCommandBuffer buffer);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanImage);

 private:
  const VkDevice& logical_device_;
  VulkanMemoryAllocationHelper& memory_alloc_helper_;

  std::tuple<VkImageType, int64_t, int64_t, int64_t, VkFormat> image_info_;
  std::pair<VkImage, VkImageView> image_and_view_;

  std::vector<int64_t> image_dims_;
  std::pair<void*, int> image_memory_;

  VkImageLayout image_layout_;
  VkAccessFlagBits image_access_flags_;
};

}  // namespace onnxruntime