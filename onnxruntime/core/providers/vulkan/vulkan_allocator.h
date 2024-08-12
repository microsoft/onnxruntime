// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

#include "vk_mem_alloc.h"  // VulkanMemoryAllocator

namespace kp {
class Manager;
}

namespace onnxruntime {
namespace vulkan {

// Device memory allocator. Assuming we also need something for host accessible memory.
class VulkanBufferAllocator : public IAllocator {
  struct Metadata {
    size_t size;
  };

 public:
  explicit VulkanBufferAllocator(OrtDevice device, VmaAllocator& manager);
  ~VulkanBufferAllocator() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanBufferAllocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  VmaAllocator& allocator_;
  VmaMemoryUsage usage_;
};

}  // namespace vulkan
}  // namespace onnxruntime
