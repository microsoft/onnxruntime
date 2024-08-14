// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "kompute/Kompute.hpp"

#include "core/framework/tensor.h"
#include "core/providers/vulkan/vulkan_memory_allocator.h"

namespace onnxruntime {
namespace vulkan {

class KomputeTensor : kp::Tensor {
 public:
  // initial creation by vulkan allocator. no tensor info available yet
  KomputeTensor(VmaAllocator allocator, uint32_t size, bool allocate_device_memory);

  void rebuild(void* /*data*/,
               uint32_t /*elementTotalCount*/,
               uint32_t /*elementMemorySize*/) override {
    // no-op for clarity.
    // if called from the base class ctor we can't do anything as allocator_ isn't set yet.
    // we create the initial staging or device buffer in the KomputeTensor ctor implementation
    // we set #elements/datatype/size later by syncing with the ORT Tensor type when available
    // which would be in VulkanDataTransfer or the OpKernel::Compute implementation after creating the Output.
  }

  // sync data type, #elements and element size from the Tensor.
  // we need to do this so the Algorithm implementation can choose good values for the workgroup.
  // TODO: how early does this need to happen?
  void SyncWithOrtTensorShape(const onnxruntime::Tensor& ort_tensor);
  // also copy data from ort_tensor. requires this to be a staging buffer.
  void SyncWithOrtTensor(const onnxruntime::Tensor& ort_tensor);

  // copy to CPU based ORT Tensor
  void CopyToOrtTensor(onnxruntime::Tensor& ort_tensor) const;

  void destroy() override;

 private:
  VmaAllocator allocator_;
  std::shared_ptr<vk::Buffer> buffer_;
  VmaAllocation allocation_;
  VmaAllocationInfo allocation_info_;
};
}  // namespace vulkan
}  // namespace onnxruntime
