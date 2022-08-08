// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_pipeline.h"

namespace onnxruntime {

// VulkanDescriptorSet methods
VulkanDescriptorSet::VulkanDescriptorSet(VkDescriptorSet descriptor_set, VkDescriptorPool descriptor_pool,
                                         const VulkanPipeline* pipeline)
    : descriptor_set_(descriptor_set),
      descriptor_pool_(descriptor_pool),
      pipeline_(pipeline) {
}

VulkanDescriptorSet::~VulkanDescriptorSet() {
  pipeline_->free_sets_.emplace_back(std::make_pair(descriptor_set_, descriptor_pool_));
}

void VulkanDescriptorSet::WriteBuffer(VkBuffer buffer, int bind_index,
                                      size_t size, VkDeviceSize offset) {
  VkWriteDescriptorSet write_set;
  ::memset(&write_set, 0, sizeof(write_set));

  VkDescriptorBufferInfo source_info;
  source_info.buffer = buffer;
  source_info.offset = offset;
  source_info.range = size;

  write_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_set.descriptorCount = 1;
  write_set.descriptorType = pipeline_->ArgType(bind_index);
  write_set.dstBinding = bind_index;
  write_set.pBufferInfo = &source_info;
  write_set.dstSet = descriptor_set_;

  VK_CALL_RETURNS_VOID(vkUpdateDescriptorSets(pipeline_->logical_device_, 1,
                                              &write_set, 0, nullptr));
}

void VulkanDescriptorSet::WriteImage(VkImageView view, VkSampler sampler,
                                     VkImageLayout layout, int bind_index) {
  VkWriteDescriptorSet write_set;
  ::memset(&write_set, 0, sizeof(write_set));

  VkDescriptorImageInfo source_info;
  source_info.imageView = view;
  source_info.imageLayout = layout;
  source_info.sampler = sampler;

  write_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_set.descriptorCount = 1;
  write_set.descriptorType = pipeline_->ArgType(bind_index);
  write_set.dstBinding = bind_index;
  write_set.pImageInfo = &source_info;
  write_set.dstSet = descriptor_set_;

  VK_CALL_RETURNS_VOID(vkUpdateDescriptorSets(pipeline_->logical_device_, 1,
                                              &write_set, 0, nullptr));
}

// VulkanPipelineFactory methods
VulkanPipelineFactory::VulkanPipelineFactory(const VkDevice& logical_device)
    : logical_device_(logical_device) {
  VkPipelineCacheCreateInfo pipeline_cache_create_info{
      /* .sType           = */ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      /* .pNext           = */ nullptr,
      /* .flags           = */ 0,  // reserved, must be 0
      /* .initialDataSize = */ 0,
      /* .pInitialData    = */ nullptr,
  };

  VK_CALL(vkCreatePipelineCache(logical_device_, &pipeline_cache_create_info, nullptr, &pipeline_cache_));

  // mStorage = std::make_shared<VulkanShaderMap>();
}

VulkanPipelineFactory::~VulkanPipelineFactory() {
  VK_CALL_RETURNS_VOID(vkDestroyPipelineCache(logical_device_, pipeline_cache_, nullptr));
}

void VulkanPipelineFactory::Reset() {
  // TODO: Avoid duplicating code between the ctor and dtor
  VK_CALL_RETURNS_VOID(vkDestroyPipelineCache(logical_device_, pipeline_cache_, nullptr));

  VkPipelineCacheCreateInfo pipeline_cache_create_info{
      /* .sType           = */ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      /* .pNext           = */ nullptr,
      /* .flags           = */ 0,  // reserved, must be 0
      /* .initialDataSize = */ 0,
      /* .pInitialData    = */ nullptr,
  };

  VK_CALL(vkCreatePipelineCache(logical_device_, &pipeline_cache_create_info, nullptr, &pipeline_cache_));
}

}  // namespace onnxruntime
