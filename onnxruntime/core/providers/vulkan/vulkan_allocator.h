// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <list>
#include <unordered_map>
#include <variant>

#include "core/framework/allocator.h"

#include "core/providers/vulkan/vulkan_utils.h"

namespace ncnn {
class VkAllocator;
}

namespace onnxruntime {
namespace vulkan {

// Device memory allocator. Assuming we also need something for host accessible memory.
class VulkanBufferAllocator : public IAllocator {
  struct Metadata {
    size_t size;
  };

 public:
  explicit VulkanBufferAllocator(OrtDevice device, ncnn::VkAllocator& allocator);
  ~VulkanBufferAllocator() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanBufferAllocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  ncnn::VkAllocator& allocator_;
};

}  // namespace vulkan
}  // namespace onnxruntime
