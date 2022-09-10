// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_memory_allocation_helper.h"
#include "vulkan_execution_provider_info.h"

#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/data_types.h"

namespace onnxruntime {

class VulkanAllocator : public IAllocator {
 public:
  VulkanAllocator(const VulkanExecutionProviderInfo& info,
                  VulkanMemoryAllocationHelper& memory_alloc_helper,
                  const VkPhysicalDeviceLimits& memory_limits);

  void* Alloc(size_t size) override;

  void* Alloc(const TensorShape& shape, MLDataType data_type) override;

  void Free(void* ptr) override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanAllocator);

 private:
  VulkanMemoryAllocationHelper& memory_alloc_helper_;
  const VkPhysicalDeviceLimits& memory_limits_;
};

}  // namespace onnxruntime