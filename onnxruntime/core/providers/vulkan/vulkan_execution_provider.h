// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "vulkan_common.h"
#include "vulkan_instance.h"
#include "vulkan_execution_provider_info.h"
#include "vulkan_execution_provider.h"
#include "vulkan_allocator.h"
#include "vulkan_sampler.h"
#include "vulkan_memory_allocation_helper.h"
#include "vulkan_command_pool.h"
#include "vulkan_pipeline.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

class VulkanExecutionProvider : public IExecutionProvider {
 public:
  explicit VulkanExecutionProvider(const VulkanExecutionProviderInfo& info);

  ~VulkanExecutionProvider();

  void RegisterAllocator(AllocatorManager& allocator_manager) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  bool ConcurrentRunSupported() const override;

  Status Sync() const override;

  const VkPhysicalDeviceLimits& GetMemoryLimits() const;

  const VulkanSampler& GetCommonSampler(bool clamp = false) const;

  VulkanMemoryAllocationHelper& GetMemoryPool() const;

  VulkanCommandPool& GetCommandPool() const;

  VulkanPipeline& GetPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                              const std::vector<uint32_t>& local_sizes = std::vector<uint32_t>()) const;

  Status QueueCommand(VkCommandBuffer cmd_buffer) const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanExecutionProvider);

 private:
  AllocatorPtr CreateVulkanAllocator();

  const VulkanExecutionProviderInfo& info_;

  std::shared_ptr<VulkanInstance> vulkan_instance_;
  VkPhysicalDevice vulkan_physical_device_;
  uint32_t vulkan_queue_family_index_;
  VkDevice vulkan_logical_device_;
  VkPhysicalDeviceProperties vulkan_device_properties_;
  VkQueue vulkan_queue_;
  VkPhysicalDeviceMemoryProperties vulkan_device_memory_properties_;

  std::shared_ptr<VulkanSampler> vulkan_sampler_;
  std::shared_ptr<VulkanSampler> vulkan_clamp_sampler_;
  std::shared_ptr<VulkanCommandPool> vulkan_command_pool_;
  std::shared_ptr<VulkanMemoryAllocationHelper> vulkan_memory_alloc_;
  std::shared_ptr<VulkanPipelineFactory> vulkan_pipeline_factory_;

  mutable std::vector<VkCommandBuffer> vulkan_command_buffers_;
};

// Registers all available Vulkan kernels
Status RegisterVulkanKernels(KernelRegistry& kernel_registry);

}  // namespace onnxruntime