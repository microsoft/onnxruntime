// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_command_pool.h"
#include "vulkan_fence.h"

namespace onnxruntime {

// VulkanCommandBuffer methods
VulkanCommandBuffer::VulkanCommandBuffer(const VkDevice& logical_device,
                                         VulkanCommandPool& command_pool) : logical_device_(logical_device),
                                                                            command_pool_(command_pool),
                                                                            buffer_(VK_NULL_HANDLE) {
  if (command_pool_.GetFreeCommandBuffers().empty()) {
    VkCommandBufferAllocateInfo command_buffer_create_info{
        /* .sType              = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        /* .pNext              = */ nullptr,
        /* .commandPool        = */ command_pool_.Get(),
        /* .level              = */ VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        /* .commandBufferCount = */ 1,
    };

    VK_CALL(vkAllocateCommandBuffers(logical_device_, &command_buffer_create_info, &buffer_));

  } else {
    auto& free_command_buffers = command_pool_.GetFreeCommandBuffers();
    auto iter = free_command_buffers.end() - 1;
    buffer_ = *iter;
    free_command_buffers.erase(iter);
  }
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
  command_pool_.GetFreeCommandBuffers().emplace_back(buffer_);
}

void VulkanCommandBuffer::BarrierSource(VkBuffer source, size_t start, size_t size, BarrierType type) const {
  VkBufferMemoryBarrier barrier;

  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.buffer = source;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.offset = start;
  barrier.pNext = nullptr;
  barrier.size = size;

  switch (type) {
    case READ_WRITE:
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
      break;
    case WRITE_WRITE:
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      break;
    default:
      break;
  }

  VK_CALL_RETURNS_VOID(vkCmdPipelineBarrier(buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                                            &barrier, 0, nullptr));
}

void VulkanCommandBuffer::Begin(VkCommandBufferUsageFlags flag) const {
  VkCommandBufferBeginInfo command_buffer_begin_info{
      /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      /* .pNext            = */ nullptr,
      /* .flags            = */ flag,
      /* .pInheritanceInfo = */ nullptr,
  };

  VK_CALL_RETURNS_VOID(vkResetCommandBuffer(buffer_, 0));

  VK_CALL(vkBeginCommandBuffer(buffer_, &command_buffer_begin_info));
}

void VulkanCommandBuffer::End() const {
  VK_CALL(vkEndCommandBuffer(buffer_));
}

// VulkanCommandPool methods
VulkanCommandPool::VulkanCommandPool(const VkDevice& logical_device,
                                     uint32_t queue_family_index) : logical_device_(logical_device),
                                                                    queue_family_index_(queue_family_index),
                                                                    command_pool_(VK_NULL_HANDLE) {
  VkCommandPoolCreateInfo command_pool_create_info{
      /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      /* .pNext            = */ nullptr,
      /* .flags            = */ VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      /* .queueFamilyIndex = */ queue_family_index_,
  };

  VK_CALL(vkCreateCommandPool(logical_device_, &command_pool_create_info, nullptr, &command_pool_));

  if (command_pool_ == VK_NULL_HANDLE) {
    ORT_THROW("Could not create Vulkan command pool");
  }
}

VulkanCommandPool::~VulkanCommandPool() {
  for (auto& cmd_buf : free_command_buffers_) {
    vkFreeCommandBuffers(logical_device_, command_pool_, 1, &cmd_buf);
  }

  vkDestroyCommandPool(logical_device_, command_pool_, nullptr);
}

void VulkanCommandPool::SubmitAndWait(VkCommandBuffer command_buffer, VkQueue queue) const {
  // TODO: Why this copy ?
  auto buf = command_buffer;

  auto fence = std::make_shared<VulkanFence>(logical_device_);

  VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              /* .pNext                = */ nullptr,
                              /* .waitSemaphoreCount   = */ 0,
                              /* .pWaitSemaphores      = */ nullptr,
                              /* .pWaitDstStageMask    = */ nullptr,
                              /* .commandBufferCount   = */ 1,
                              /* .pCommandBuffers      = */ &buf,
                              /* .signalSemaphoreCount = */ 0,
                              /* .pSignalSemaphores    = */ nullptr};

  VK_CALL(vkQueueSubmit(queue, 1, &submit_info, fence->Get()));

  fence->Wait();
}

VulkanCommandBuffer* VulkanCommandPool::AllocBuffer() {
  return new VulkanCommandBuffer(logical_device_, *this);
}

}  // namespace onnxruntime
