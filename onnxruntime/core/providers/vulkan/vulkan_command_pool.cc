// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_command_pool.h"

namespace onnxruntime {

// VulkanCommandBuffer methods
VulkanCommandBuffer::VulkanCommandBuffer(const VulkanCommandPool& vulkan_command_pool) : vulkan_command_pool_(vulkan_command_pool) {
  if (vulkan_command_pool_.free_vulkan_command_buffers_.empty()) {
    VkCommandBufferAllocateInfo cmdBufferCreateInfo{
        /* .sType              = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        /* .pNext              = */ nullptr,
        /* .commandPool        = */ cmdPool,
        /* .level              = */ level,
        /* .commandBufferCount = */ cmdBufferCount,
    };

    VK_CALL(vkAllocateCommandBuffers(mDevice, &cmdBufferCreateInfo, cmdBuffers));
  } else {
    auto iter = pool->mFreeBuffers.end() - 1;
    mBuffer = *iter;
    pool->mFreeBuffers.erase(iter);
  }
}
VulkanCommandPool::Buffer::~Buffer() {
  mPool->mFreeBuffers.emplace_back(mBuffer);
}

void VulkanCommandPool::Buffer::barrierSource(VkBuffer source, size_t start, size_t size, BarrierType type) const {
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
  vkCmdPipelineBarrier(mBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                       &barrier, 0, nullptr);
}
void VulkanCommandPool::Buffer::begin(VkCommandBufferUsageFlags flag) const {
  VkCommandBufferBeginInfo cmdBufferBeginInfo{
      /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      /* .pNext            = */ nullptr,
      /* .flags            = */ flag,
      /* .pInheritanceInfo = */ nullptr,
  };
  vkResetCommandBuffer(mBuffer, 0);
  CALL_VK(vkBeginCommandBuffer(mBuffer, &cmdBufferBeginInfo));
}
void VulkanCommandPool::Buffer::end() const {
  CALL_VK(vkEndCommandBuffer(mBuffer));
}

// VulkanCommandPool methods
VulkanCommandPool::VulkanCommandPool(const VkDevice& vulkan_logical_device,
                                     uint32_t vulkan_queue_family_index) : vulkan_logical_device_(vulkan_logical_device),
                                                                           vulkan_command_pool_(VK_NULL_HANDLE) {
  VkCommandPoolCreateInfo vulkan_command_pool_create_info{
      /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      /* .pNext            = */ nullptr,
      /* .flags            = */ VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      /* .queueFamilyIndex = */ vulkan_queue_family_index,
  };

  VK_CALL(vkCreateCommandPool(vulkan_logical_device_, &vulkan_command_pool_create_info, nullptr, &vulkan_command_pool_));

  if (vulkan_command_pool_ == VK_NULL_HANDLE) {
    ORT_THROW("Could not create Vulkan command pool");
  }
}

VulkanCommandPool::~VulkanCommandPool() {
  for (auto& cmd_buf : free_vulkan_command_buffers_) {
    vkFreeCommandBuffers(vulkan_logical_device_, vulkan_command_pool_, 1, &cmd_buf);
  }

  vkDestroyCommandPool(vulkan_logical_device_, vulkan_command_pool_, nullptr);
}

void VulkanCommandPool::SubmitAndWait(VkCommandBuffer cmd_buf, VkQueue vulkan_queue) const {
  // TODO: Why this copy ?
  auto buf = cmd_buf;

  // TODO: Once VulkanFence is supported, this should go away
  auto fence = std::make_shared<VulkanFence>(mDevice);
  VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              /* .pNext                = */ nullptr,
                              /* .waitSemaphoreCount   = */ 0,
                              /* .pWaitSemaphores      = */ nullptr,
                              /* .pWaitDstStageMask    = */ nullptr,
                              /* .commandBufferCount   = */ 1,
                              /* .pCommandBuffers      = */ &buf,
                              /* .signalSemaphoreCount = */ 0,
                              /* .pSignalSemaphores    = */ nullptr};
  auto queue = mDevice.acquireDefaultDevQueue();
  VK_CALL(vkQueueSubmit(vulkan_queue, 1, &submit_info, fence->Get()));
  fence->Wait();
}

VulkanCommandBuffer* VulkanCommandPool::AllocBuffer() const {
  return new VulkanCommandBuffer(*this);
}

}  // namespace onnxruntime
