// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_memory_pool.h"

namespace onnxruntime {

class VulkanImage {
 public:
  VulkanImage(const VulkanMemoryPool& pool, bool seperate, const std::vector<int>& dims,
              halide_type_t type = halide_type_of<float>());
  VulkanImage(const VulkanMemoryPool& pool, bool seperate, int w, int h)
      : VulkanImage(pool, seperate, std::vector<int>{w, h}) {
  }
  virtual ~VulkanImage();

  inline int width() const {
    return std::get<1>(mInfo);
  }
  inline int height() const {
    return std::get<2>(mInfo);
  }
  inline int depth() const {
    return std::get<3>(mInfo);
  }
  inline std::vector<int> dims() const {
    return mDims;
  }
  inline VkImage get() const {
    return mImage.first;
  }
  inline VkImageView view() const {
    return mImage.second;
  }
  inline VkFormat format() const {
    return std::get<4>(mInfo);
  }
  void release();
  void resetBarrier() {
    mLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  }
  VkImageLayout currentLayout() const {
    return mLayout;
  }
  void barrierWrite(VkCommandBuffer buffer) const;
  void barrierRead(VkCommandBuffer buffer) const;

 private:
  std::tuple<VkImageType, uint32_t, uint32_t, uint32_t, VkFormat> mInfo;
  std::pair<VkImage, VkImageView> mImage;
  const VulkanDevice& mDevice;
  std::vector<int> mDims;
  const VulkanMemoryPool& mPool;
  std::pair<void*, int> mMemory;
  mutable VkImageLayout mLayout;
  mutable VkAccessFlagBits mAccess;
};
}  // namespace onnxruntime