// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

#include <unordered_map>
#include <list>
#include <functional>
#include <variant>
#include "vulkan_utils.h"

namespace onnxruntime {
namespace vulkan {

// Device memory allocator. Assuming we also need something for host accessible memory.
class VulkanBufferAllocator : public IAllocator {
  struct Metadata {
    size_t size;
  };

 public:
  explicit VulkanBufferAllocator();
  ~VulkanBufferAllocator() = default;

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
};

}  // namespace vulkan
}  // namespace onnxruntime
