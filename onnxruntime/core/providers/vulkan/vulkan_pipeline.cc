// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_pipeline.h"

namespace onnxruntime {

// VulkanDescriptorSet methods
VulkanDescriptorSet::VulkanDescriptorSet(const VkDevice& logical_device,
                                         VkDescriptorSet descriptor_set, VkDescriptorPool descriptor_pool,
                                         VulkanPipeline* pipeline)
    : logical_device_(logical_device),
      descriptor_set_(descriptor_set),
      descriptor_pool_(descriptor_pool),
      pipeline_(pipeline) {
}

VulkanDescriptorSet::~VulkanDescriptorSet() {
  pipeline_->GetFreeDescriptorSets().emplace_back(std::make_pair(descriptor_set_, descriptor_pool_));
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

  VK_CALL_RETURNS_VOID(vkUpdateDescriptorSets(logical_device_, 1,
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

  VK_CALL_RETURNS_VOID(vkUpdateDescriptorSets(logical_device_, 1,
                                              &write_set, 0, nullptr));
}

// VulkanPipeline methods
VulkanPipeline::VulkanPipeline(const VkDevice& logical_device, VkPipeline pipeline,
                               VkPipelineLayout layout, const std::vector<VkDescriptorPoolSize>& descriptor_pool_sizes,
                               VkDescriptorSetLayout descriptor_set_layout, const std::vector<VkDescriptorType>& descriptor_types)
    : logical_device_(logical_device),
      pipeline_(pipeline),
      pipeline_layout_(layout),
      descriptor_pool_sizes_(descriptor_pool_sizes),
      descriptor_set_layout_(descriptor_set_layout),
      descriptor_types_(descriptor_types) {
  // Nothing to do here
}

VulkanPipeline* VulkanPipeline::Create(const VkDevice& logical_device, const uint8_t* /*data*/, size_t /*length*/,
                                       const std::vector<VkDescriptorType>& descriptor_types, VkPipelineCache pipleine_cache,
                                       const std::vector<uint32_t>& local_size) {
  // TODO: Fix me
  VkShaderModule shader_out = VK_NULL_HANDLE;
  //VkResult result = dev.createShaderModule(shaderOut, length, (const uint32_t*)data);

  //if (VK_SUCCESS != result) {
  //  return nullptr;
  //}

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  std::unordered_map<VkDescriptorType, int> type_count;

  for (size_t i = 0; i < descriptor_types.size(); ++i) {
    auto type = descriptor_types[i];

    if (type_count.find(type) == type_count.end()) {
      type_count[type] = 1;
    } else {
      type_count[type] += 1;
    }

    VkDescriptorSetLayoutBinding binding;

    binding.binding = static_cast<uint32_t>(i);
    binding.descriptorType = type;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.pImmutableSamplers = nullptr;

    bindings.emplace_back(binding);
  }

  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;

  {
    // VkDescriptorSetLayout creation
    VkDescriptorSetLayoutCreateInfo set_layout_info;
    set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    set_layout_info.pBindings = bindings.data();
    set_layout_info.pNext = nullptr;
    set_layout_info.flags = 0;
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

    VK_CALL(vkCreateDescriptorSetLayout(logical_device, &set_layout_info, nullptr, &set_layout));

    // VkPipelineLayout creation
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    VK_CALL(vkCreatePipelineLayout(logical_device, &pipeline_layout_info, nullptr, &pipeline_layout));
  }

  std::vector<VkSpecializationMapEntry> specialization_map_entry;
  std::shared_ptr<VkSpecializationInfo> specialization_info = std::make_shared<VkSpecializationInfo>();

  if (local_size.size() > 0) {
    for (size_t i = 0; i < local_size.size(); i++) {
      VkSpecializationMapEntry entry = {static_cast<uint32_t>(i + 1), static_cast<uint32_t>(sizeof(uint32_t) * i),
                                        sizeof(uint32_t)};
      specialization_map_entry.push_back(entry);
    }

    specialization_info->pData = local_size.data();
    specialization_info->dataSize = local_size.size() * sizeof(uint32_t);
    specialization_info->pMapEntries = specialization_map_entry.data();
    specialization_info->mapEntryCount = static_cast<uint32_t>(specialization_map_entry.size());
  }

  // Create the pipeline cache
  VkPipeline pipeline;

  VkComputePipelineCreateInfo pipeline_info;
  ::memset(&pipeline_info, 0, sizeof(pipeline_info));

  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = shader_out;
  pipeline_info.stage.pName = "main";
  pipeline_info.layout = pipeline_layout;
  pipeline_info.stage.pSpecializationInfo = specialization_info.get();

  auto res = vkCreateComputePipelines(logical_device, pipleine_cache, 1, &pipeline_info, nullptr, &pipeline);

  if (VK_SUCCESS != res) {
    VK_CALL_RETURNS_VOID(vkDestroyShaderModule(logical_device, shader_out, nullptr));
    VK_CALL_RETURNS_VOID(vkDestroyPipelineLayout(logical_device, pipeline_layout, nullptr));
    VK_CALL_RETURNS_VOID(vkDestroyDescriptorSetLayout(logical_device, set_layout, nullptr));

    return nullptr;
  }

  VK_CALL_RETURNS_VOID(vkDestroyShaderModule(logical_device, shader_out, nullptr));

  std::vector<VkDescriptorPoolSize> descriptor_pool_size;
  for (auto& iter : type_count) {
    VkDescriptorPoolSize s;
    s.descriptorCount = iter.second;
    s.type = iter.first;

    descriptor_pool_size.emplace_back(s);
  }

  return new VulkanPipeline(logical_device, pipeline, pipeline_layout, descriptor_pool_size, set_layout, descriptor_types);
}

VulkanPipeline::~VulkanPipeline() {
  for (auto& iter : free_sets_) {
    try {
      VK_CALL(vkFreeDescriptorSets(logical_device_, iter.second, 1, &iter.first));
    } catch (const std::exception& /*ex*/) {
      // TODO: Log this
    }

    VK_CALL_RETURNS_VOID(vkDestroyDescriptorPool(logical_device_, iter.second, nullptr));
  }

  VK_CALL_RETURNS_VOID(vkDestroyPipelineLayout(logical_device_, pipeline_layout_, nullptr));
  VK_CALL_RETURNS_VOID(vkDestroyDescriptorSetLayout(logical_device_, descriptor_set_layout_, nullptr));
  VK_CALL_RETURNS_VOID(vkDestroyPipeline(logical_device_, pipeline_, nullptr));
}

void VulkanPipeline::Bind(VkCommandBuffer buffer, VkDescriptorSet descriptor_set) const {
  // Bind the compute pipeline
  VK_CALL_RETURNS_VOID(vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_));

  // Bind descriptor set
  VK_CALL_RETURNS_VOID(vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                               pipeline_layout_, 0, 1, &descriptor_set, 0, nullptr));
}

VulkanDescriptorSet* VulkanPipeline::CreateSet() {
  if (!free_sets_.empty()) {
    auto iter = free_sets_.end() - 1;
    auto res = new VulkanDescriptorSet(logical_device_, iter->first, iter->second, this);
    free_sets_.erase(iter);
    return res;
  }

  // Descriptor pool
  VkDescriptorPool descriptor_pool;

  VkDescriptorPoolCreateInfo descriptor_pool_create_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptor_pool_create_info.poolSizeCount = static_cast<uint32_t>(descriptor_pool_sizes_.size());
  descriptor_pool_create_info.pPoolSizes = descriptor_pool_sizes_.data();
  descriptor_pool_create_info.maxSets = 1;
  descriptor_pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  VK_CALL(vkCreateDescriptorPool(logical_device_, &descriptor_pool_create_info, nullptr, &descriptor_pool));

  // Descriptor set
  VkDescriptorSet descriptor_set;

  VkDescriptorSetAllocateInfo descriptor_set_allocate_set;
  ::memset(&descriptor_set_allocate_set, 0, sizeof(descriptor_set_allocate_set));

  descriptor_set_allocate_set.pNext = nullptr;
  descriptor_set_allocate_set.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_set_allocate_set.descriptorPool = descriptor_pool;
  descriptor_set_allocate_set.descriptorSetCount = 1;
  descriptor_set_allocate_set.pSetLayouts = &descriptor_set_layout_;

  VK_CALL(vkAllocateDescriptorSets(logical_device_, &descriptor_set_allocate_set, &descriptor_set));

  return new VulkanDescriptorSet(logical_device_, descriptor_set, descriptor_pool, this);
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

  // TODO: Fix once we have the concept of ShaderMap
  // mStorage = std::make_shared<VulkanShaderMap>();
}

VulkanPipelineFactory::~VulkanPipelineFactory() {
  VK_CALL_RETURNS_VOID(vkDestroyPipelineCache(logical_device_, pipeline_cache_, nullptr));
}

void VulkanPipelineFactory::Reset() {
  // TODO: Avoid duplicating code in the VulkanPipelineFactory ctor and dtor
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

VulkanPipeline* VulkanPipelineFactory::GetPipeline(const std::string& key,
                                                         const std::vector<VkDescriptorType>& /*descriptor_types*/,
                                                         const std::vector<uint32_t>& /*local_sizes*/) {
  auto iter = pipelines_.find(key);
  if (iter != pipelines_.end()) {
    return iter->second.get();
  }

  // TODO: Fix once we have the concept of ShaderMap
  /*
  auto content = mStorage->search(key);

  if (nullptr == content.first) {
    return nullptr;
  }

  auto* pipeline = VulkanPipeline::Create(logical_device_, content.first, content.second,
                                          descriptor_types, pipeline_cache_, local_sizes);

  ORT_ENFORCE(pipeline, "Could not create a Vulkan pipeline for ", key);

  pipelines_.insert(std::make_pair(key, std::shared_ptr<VulkanPipeline>(pipeline)));

  return pipeline;
  */

  return nullptr;
}

}  // namespace onnxruntime
