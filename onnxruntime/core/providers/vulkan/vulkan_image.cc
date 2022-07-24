// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_image.h"

namespace onnxruntime {

VulkanImage::VulkanImage(VulkanMemoryPool& vulkan_memory_pool, bool seperate, const std::vector<int64_t>& vulkan_image_dims, MLDataType data_type)
    : vulkan_logical_device_(vulkan_memory_pool.GetLogicalDevice()), vulkan_memory_pool_(vulkan_memory_pool) {
  if (data_type != DataTypeImpl::GetTensorType<float>()) {
    ORT_THROW("Only creating float Vulkan images is currently supported");
  }

  auto vulkan_image_dims_size = vulkan_image_dims.size();

  if (!(vulkan_image_dims_size >= 1 && vulkan_image_dims_size <= 3)) {
    ORT_THROW("Only 1D, 2D, or 3D Vulkan images are supported for now");
  }

  auto image_type = VK_IMAGE_TYPE_1D;
  auto view_type = VK_IMAGE_VIEW_TYPE_1D;

  vulkan_image_dims_ = vulkan_image_dims;

  auto image_width = vulkan_image_dims_[0];
  auto image_height = 1;
  auto image_depth = 1;

  if (vulkan_image_dims_size > 1) {
    image_height = vulkan_image_dims_[1];
    image_type = VK_IMAGE_TYPE_2D;
    view_type = VK_IMAGE_VIEW_TYPE_2D;
  }

  if (vulkan_image_dims_size > 2) {
    image_depth = vulkan_image_dims_[2];
    image_type = VK_IMAGE_TYPE_3D;
    view_type = VK_IMAGE_VIEW_TYPE_3D;
  }

  auto image_format = VK_FORMAT_R32G32B32A32_SFLOAT;

  vulkan_image_info_ = std::make_tuple(image_type, image_width, image_height, image_depth, image_format);

  vulkan_image_and_view_.first = vulkan_memory_pool_.allocImage(vulkan_image_info_);

  vulkan_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  vulkan_image_access_flags_ = VK_ACCESS_SHADER_READ_BIT;

  VkMemoryRequirements image_memory_requirements;

  mDevice.getImageMemoryRequirements(vulkan_image_and_view_.first, image_memory_requirements);

  vulkan_image_memory_ = vulkan_memory_pool_.Alloc(image_memory_requirements, 0, seperate);

  auto realMem = static_cast<VulkanMemory*>(vulkan_image_memory_.first);

  mDevice.bindImageMemory(mImage.first, realMem->get(), mMemory.second);

  CALL_VK(mDevice.createImageView(mImage.second, mImage.first, viewType, format));
}

VulkanImage::~VulkanImage() {
  mDevice.destroyImageView(mImage.second, nullptr);
  vulkan_memory_pool_.returnImage(std::move(vulkan_image_and_view_.first), std::move(vulkan_image_info_));
  if (nullptr != vulkan_image_memory_.first) {
    vulkan_memory_pool_.Free(vulkan_image_memory_);
  }
}

void VulkanImage::Release() {
  if (nullptr == vulkan_image_memory_.first) {
    return;
  }

  vulkan_memory_pool_.Free(vulkan_image_memory_);

  vulkan_image_memory_.first = nullptr;
}

void VulkanImage::BarrierWrite(VkCommandBuffer buffer) const {
  VkImageMemoryBarrier barrier;
  ::memset(&barrier, 0, sizeof(VkImageMemoryBarrier));

  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask = vulkan_image_access_flags_;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.image = vulkan_image_and_view_.first;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.oldLayout = vulkan_image_layout_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  VK_CALL_RETURNS_VOID(vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                            nullptr, 0, nullptr, 1, &barrier));

  vulkan_image_layout_ = VK_IMAGE_LAYOUT_GENERAL;
  vulkan_image_access_flags_ = VK_ACCESS_SHADER_WRITE_BIT;
}

void VulkanImage::BarrierRead(VkCommandBuffer buffer) const {
  if (vulkan_image_access_flags_ == VK_ACCESS_SHADER_READ_BIT && vulkan_image_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    return;
  }

  VkImageMemoryBarrier barrier;
  ::memset(&barrier, 0, sizeof(VkImageMemoryBarrier));

  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask = vulkan_image_access_flags_;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.image = vulkan_image_and_view_.first;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.oldLayout = vulkan_image_layout_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
  vulkan_image_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vulkan_image_access_flags_ = VK_ACCESS_SHADER_READ_BIT;
}

}  // namespace onnxruntime
