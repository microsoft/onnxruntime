// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_image.h"

namespace onnxruntime {

static VkResult CreateImageView(const VkDevice& logical_device, VkImageView& image_view, const VkImage& image, const VkImageViewType& view_type,
                                const VkFormat& format) {
  VkImageViewCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.image = image;
  info.viewType = view_type;
  info.format = format;
  info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;

  return vkCreateImageView(logical_device, &info, nullptr, &image_view);
}

VulkanImage::VulkanImage(VulkanMemoryPool& memory_pool, const std::vector<int64_t>& image_dims,
                         MLDataType data_type)
    : logical_device_(memory_pool.GetLogicalDevice()), memory_pool_(memory_pool) {
  if (data_type != DataTypeImpl::GetTensorType<float>()) {
    ORT_THROW("Only creating float Vulkan images is currently supported");
  }

  size_t image_dims_size = image_dims.size();

  if (!(image_dims_size >= 1 && image_dims_size <= 3)) {
    ORT_THROW("Only 1D, 2D, or 3D Vulkan images are supported for now");
  }

  // Identify VkImage metadata 
  VkImageType image_type = VK_IMAGE_TYPE_1D;
  VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_1D;

  image_dims_ = image_dims;

  int64_t image_width = image_dims_[0];
  int64_t image_height = 1;
  int64_t image_depth = 1;

  if (image_dims_size > 1) {
    image_height = image_dims_[1];
    image_type = VK_IMAGE_TYPE_2D;
    view_type = VK_IMAGE_VIEW_TYPE_2D;
  }

  if (image_dims_size > 2) {
    image_depth = image_dims_[2];
    image_type = VK_IMAGE_TYPE_3D;
    view_type = VK_IMAGE_VIEW_TYPE_3D;
  }

  VkFormat image_format = VK_FORMAT_R32G32B32A32_SFLOAT;

  image_info_ = std::make_tuple(image_type, image_width, image_height, image_depth, image_format);

  // Allocate VkImage
  image_and_view_.first = memory_pool_.AllocVkImage(image_info_);

  image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  image_access_flags_ = VK_ACCESS_SHADER_READ_BIT;

  // Identify VkImage memory requirements
  VkMemoryRequirements image_memory_requirements;

  VK_CALL_RETURNS_VOID(vkGetImageMemoryRequirements(logical_device_, image_and_view_.first, &image_memory_requirements));

  // Allocate the necessary memory
  image_memory_ = memory_pool_.Alloc(image_memory_requirements, 0);

  auto* vulkan_memory = static_cast<VulkanMemory*>(image_memory_.first);

  // Bind device memory to VkImage
  VK_CALL(vkBindImageMemory(logical_device_, image_and_view_.first, vulkan_memory->Get(), image_memory_.second));

  // Create VkImageView
  VK_CALL(CreateImageView(logical_device_, image_and_view_.second, image_and_view_.first, view_type, image_format));
}

VulkanImage::~VulkanImage() {
    // Free VkImageView first
  VK_CALL_RETURNS_VOID(vkDestroyImageView(logical_device_, image_and_view_.second, nullptr));

  // Free the VkImage via the memory pool
  memory_pool_.FreeVkImage(std::move(image_and_view_.first));

  // Free the actual device memory backing the VkImage
  if (nullptr != image_memory_.first) {
    memory_pool_.Free(image_memory_);
  }
}

void VulkanImage::Release() {
  if (nullptr == image_memory_.first) {
    return;
  }

  // Free the image
  memory_pool_.Free(image_memory_);

  image_memory_.first = nullptr;

  // TODO: Should the image view be cleaned up too ?
}

void VulkanImage::BarrierWrite(VkCommandBuffer buffer) {
  VkImageMemoryBarrier barrier;
  ::memset(&barrier, 0, sizeof(VkImageMemoryBarrier));

  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask = image_access_flags_;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.image = image_and_view_.first;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.oldLayout = image_layout_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  VK_CALL_RETURNS_VOID(vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                            nullptr, 0, nullptr, 1, &barrier));

  image_layout_ = VK_IMAGE_LAYOUT_GENERAL;
  image_access_flags_ = VK_ACCESS_SHADER_WRITE_BIT;
}

void VulkanImage::BarrierRead(VkCommandBuffer buffer) {
  if (image_access_flags_ == VK_ACCESS_SHADER_READ_BIT && image_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    return;
  }

  VkImageMemoryBarrier barrier;
  ::memset(&barrier, 0, sizeof(VkImageMemoryBarrier));

  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask = image_access_flags_;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.image = image_and_view_.first;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.oldLayout = image_layout_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  VK_CALL_RETURNS_VOID(vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                            nullptr, 0, nullptr, 1, &barrier));

  image_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_access_flags_ = VK_ACCESS_SHADER_READ_BIT;
}

}  // namespace onnxruntime
