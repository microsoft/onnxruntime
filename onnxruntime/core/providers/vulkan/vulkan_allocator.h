// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//
// TODO: Not currently used as the implementation is creating a compiled model. Required to use static kernels.
//

#include "core/framework/allocator.h"

#define VULKAN_EP_USE_VMA
#ifdef VULKAN_EP_USE_VMA
#include "vk_mem_alloc.h"  // VulkanMemoryAllocator
#endif

namespace kp {
class Manager;
}

namespace onnxruntime {
namespace vulkan {

// Device memory allocator. Assuming we also need something for host accessible memory.
#ifdef VULKAN_EP_USE_VMA
class VulkanBufferAllocator : public IAllocator {
  struct Metadata {
    size_t size;
  };

 public:
  explicit VulkanBufferAllocator(OrtDevice device, VmaAllocator& allocator);
  ~VulkanBufferAllocator() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanBufferAllocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  VmaAllocator& allocator_;
  const bool allocate_device_memory_;  // if false, allocate staging memory
};
#else
class VulkanBufferAllocator : public IAllocator {
  struct Metadata {
    size_t size;
  };

 public:
  explicit VulkanBufferAllocator(OrtDevice device, kp::Manager& manager);
  ~VulkanBufferAllocator() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanBufferAllocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  kp::Manager& manager_;
};

#endif

}  // namespace vulkan
}  // namespace onnxruntime
