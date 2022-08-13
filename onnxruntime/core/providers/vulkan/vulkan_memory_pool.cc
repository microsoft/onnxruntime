// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_memory_pool.h"

#include "core/common/common.h"

namespace onnxruntime {

// VulkanAllocator methods
VulkanAllocator::VulkanAllocator(const VkDevice& logical_device, size_t memory_type_index)
    : IAllocator(OrtMemoryInfo(VULKAN, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0),
                               0, OrtMemTypeDefault)),  // TODO: Should the device id be made configurable ?
      logical_device_(logical_device),
      memory_type_index_(memory_type_index) {}

void* VulkanAllocator::Alloc(size_t size) {
  VkMemoryAllocateInfo info;

  info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  info.pNext = nullptr;
  info.allocationSize = size;
  info.memoryTypeIndex = static_cast<uint32_t>(memory_type_index_);

  return new VulkanMemory(logical_device_, info);
}

void VulkanAllocator::Free(void* ptr) {
  delete static_cast<VulkanMemory*>(ptr);
}

// VulkanMemory methods
VulkanMemory::VulkanMemory(const VkDevice& logical_device, const VkMemoryAllocateInfo& info) : logical_device_(logical_device) {
  VK_CALL(vkAllocateMemory(logical_device_, &info, nullptr, &memory_));
  type_index_ = info.memoryTypeIndex;
  size_ = info.allocationSize;
}

VulkanMemory::~VulkanMemory() {
  VK_CALL_RETURNS_VOID(vkFreeMemory(logical_device_, memory_, nullptr));
}

// VulkanMemoryPool methods
VulkanMemoryPool::VulkanMemoryPool(const VkDevice& logical_device,
                                   VkPhysicalDeviceMemoryProperties physical_device_memory_props,
                                   uint32_t queue_family_index) : logical_device_(logical_device),
                                                                  physical_device_memory_props_(physical_device_memory_props),
                                                                  queue_family_index_(queue_family_index) {
  size_t memory_type_count = static_cast<size_t>(physical_device_memory_props.memoryTypeCount);

  allocators_.reserve(memory_type_count);

  for (size_t i = 0; i < memory_type_count; ++i) {
    allocators_.push_back(std::make_unique<VulkanAllocator>(logical_device_, i));
  }
}

std::pair<void*, int> VulkanMemoryPool::Alloc(const VkMemoryRequirements& requirements, VkFlags alloc_flags) {
  // TODO: Implement an actual memory pool to cache VulkanMemory
  // for re-use. Right now, we are making an actual device allocation
  // for every Alloc() call which is very costly

  size_t index = 0;

  uint32_t type_bits = requirements.memoryTypeBits;

  for (size_t i = 0; i < physical_device_memory_props_.memoryTypeCount; i++) {
    // TODO: Do we need the following if ?
    if ((type_bits & 1) == 1) {
      if ((physical_device_memory_props_.memoryTypes[i].propertyFlags & alloc_flags) == alloc_flags) {
        index = i;
        break;
      }
    }

    type_bits >>= 1;
  }

  ORT_ENFORCE(index >= 0 && index < allocators_.size());

  // TODO: Account for requirements.alignment ?
  auto* vulkan_memory = static_cast<VulkanMemory*>(allocators_[index]->Alloc(requirements.size));

  return std::make_pair(vulkan_memory, 0);
}

void VulkanMemoryPool::Free(std::pair<void*, int>& memory) {
  auto* vulkan_memory = static_cast<VulkanMemory*>(memory.first);

  allocators_[vulkan_memory->Type()]->Free(vulkan_memory);

  return;
}

VkBuffer VulkanMemoryPool::AllocVkBuffer(size_t size, VkBufferUsageFlags flags, VkSharingMode shared) {
  VkBuffer buffer;

  VkBufferCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;
  info.size = static_cast<VkDeviceSize>(size);
  info.usage = flags;
  info.sharingMode = shared;
  info.pQueueFamilyIndices = &queue_family_index_;
  info.queueFamilyIndexCount = 1;

  VK_CALL(vkCreateBuffer(logical_device_, &info, nullptr, &buffer));

  return buffer;
}

void VulkanMemoryPool::FreeVkBuffer(VkBuffer buffer) {
  VK_CALL_RETURNS_VOID(vkDestroyBuffer(logical_device_, buffer, nullptr));
}

VkImage VulkanMemoryPool::AllocVkImage(const std::tuple<VkImageType, int64_t, int64_t, int64_t, VkFormat>& image_info) {
  VkImage image;

  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = std::get<0>(image_info);
  info.extent.width = static_cast<uint32_t>(std::get<1>(image_info));
  info.extent.height = static_cast<uint32_t>(std::get<2>(image_info));
  info.extent.depth = static_cast<uint32_t>(std::get<3>(image_info));
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.format = std::get<4>(image_info);
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.pNext = nullptr;

  VK_CALL(vkCreateImage(logical_device_, &info, nullptr, &image));

  return image;
}

void VulkanMemoryPool::FreeVkImage(VkImage image) {
  VK_CALL_RETURNS_VOID(vkDestroyImage(logical_device_, image, nullptr));
}

}  // namespace onnxruntime
