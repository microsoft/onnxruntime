// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_memory_allocation_helper.h"
#include "vulkan_image.h"

#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

class VulkanTensor {
 public:
  VulkanTensor(const TensorShape& tensor_shape, MLDataType data_type,
               VulkanMemoryAllocationHelper& memory_alloc_helper,
               const VkPhysicalDeviceLimits& memory_limits);

  virtual ~VulkanTensor() = default;

  size_t NumImages() const {
    return images_.size();
  }

  const std::array<int64_t, 2>& Blocks() const {
    return blocks_;
  }

  const VulkanImage* Get(size_t index = 0) const {
    return images_[index].get();
  }

  // N, H, W, C
  static std::array<int64_t, 4> TensorShapeFormat(const TensorShape& tensor_shape);

  static int64_t GetAlignSize(const Tensor* tensor);

  void Release();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanTensor);

 private:
  std::vector<std::shared_ptr<VulkanImage>> images_;
  std::array<int64_t, 2> blocks_;
  std::array<int64_t, 4> size_;
};

}  // namespace onnxruntime