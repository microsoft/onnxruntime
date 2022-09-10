// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_execution_provider_info.h"
#include "vulkan_execution_provider.h"
#include "vulkan_allocator.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

class VulkanInstance {
 public:
  VulkanInstance();
  ~VulkanInstance();
  VkInstance Get() const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanInstance);

 private:
  VkInstance vulkan_instance_;
};

class VulkanExecutionProvider : public IExecutionProvider {
  explicit VulkanExecutionProvider(const VulkanExecutionProviderInfo& info);

  ~VulkanExecutionProvider();

  void RegisterAllocator(AllocatorManager& allocator_manager) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

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
};

// Registers all available Vulkan kernels
Status RegisterVulkanKernels(KernelRegistry& kernel_registry);

}  // namespace onnxruntime