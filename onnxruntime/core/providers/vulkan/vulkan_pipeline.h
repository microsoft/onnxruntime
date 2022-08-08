// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

namespace onnxruntime {

class VulkanPipeline;

class VulkanDescriptorSet {
 public:
  VulkanDescriptorSet(VkDescriptorSet descriptor_set, VkDescriptorPool descriptor_pool,
                      const VulkanPipeline* pipeline);

  virtual ~VulkanDescriptorSet();

  void WriteBuffer(VkBuffer buffer, int bind_index, size_t size, VkDeviceSize offset = 0);

  void WriteImage(VkImageView view, VkSampler sampler, VkImageLayout layout, int bind_index);

  VkDescriptorSet Get() const {
    return descriptor_set_;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanDescriptorSet);

 private:
  VkDescriptorSet descriptor_set_;
  VkDescriptorPool descriptor_pool_;
  const VulkanPipeline* pipeline_;
};

class VulkanPipeline {
 public:
  static VulkanPipeline* Create(const VkDevice& vulkan_logical_device, const uint8_t* data, size_t length,
                                const std::vector<VkDescriptorType>& buffer_types, VkPipelineCache cache,
                                const std::vector<uint32_t>& local_size = std::vector<uint32_t>());
  virtual ~VulkanPipeline();

  VkPipeline Get() const {
    return pipeline_;
  }

  void Bind(VkCommandBuffer buffer, VkDescriptorSet descriptor_set) const;

  inline VkDescriptorType ArgType(int index) const {
    return buffer_types_[index];
  }

  VulkanDescriptorSet* CreateSet() const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanPipeline);

 private:
  VulkanPipeline(const VkDevice& logical_device, VkPipeline pipeline,
                 VkPipelineLayout layout, const std::vector<VkDescriptorPoolSize>& descriptor_pool_sizes,
                 VkDescriptorSetLayout descriptor_set_layout, const std::vector<VkDescriptorType>& buffer_types);

  const VkDevice& logical_device_;
  VkPipeline pipeline_;
  VkPipelineLayout pipleine_layout_;
  std::vector<VkDescriptorPoolSize> descriptor_pool_sizes_;
  VkDescriptorSetLayout descriptor_set_layout_;
  std::vector<VkDescriptorType> buffer_types_;
  mutable std::vector<std::pair<VkDescriptorSet, VkDescriptorPool>> free_sets_;

  friend class VulkanDescriptorSet;
};

class VulkanPipelineFactory {
 public:
  explicit VulkanPipelineFactory(const VkDevice& logical_device);

  virtual ~VulkanPipelineFactory();

  const VulkanPipeline* GetPipeline(const std::string& key,
                                    const std::vector<VkDescriptorType>& descriptor_types,
                                    const std::vector<uint32_t>& local_sizes = std::vector<uint32_t>()) const;

  void Reset();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanPipelineFactory);

 private:
  const VkDevice& logical_device_;
  mutable std::unordered_map<std::string, std::shared_ptr<VulkanPipeline>> pipelines_;
  VkPipelineCache pipeline_cache_;

  // std::shared_ptr<VulkanShaderMap> shader_map_;
};
}  // namespace onnxruntime