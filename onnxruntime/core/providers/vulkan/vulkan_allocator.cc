// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// TODO: Not currently used as the implementation is creating a compiled model. Required to use static kernels.
//

#include "core/providers/vulkan/vulkan_allocator.h"

#include <iostream>

#include "kompute/Manager.hpp"

#include "core/framework/ortdevice.h"
#include "core/providers/vulkan/ort_kompute_tensor.h"

namespace onnxruntime {
namespace vulkan {

namespace {
OrtMemoryInfo CreateMemoryInfo(OrtDevice device) {
  // TODO: If this in integrated GPU we should be able to use device memory for both.
  // May not matter here though as it might be handled in the implementation of MemcpyToHost
  if (device.MemType() == OrtDevice::MemType::DEFAULT) {
    return OrtMemoryInfo("VulkanAllocator",
                         OrtAllocatorType::OrtDeviceAllocator,
                         device,
                         /*id*/ 0,
                         OrtMemTypeDefault);
  } else {
    return OrtMemoryInfo("VulkanStagingAllocator",
                         OrtAllocatorType::OrtDeviceAllocator,
                         device,
                         /*id*/ 0,
                         OrtMemTypeCPU);
  };
}
}  // namespace

// This doesn't work with Kompute as you can't plugin an existing allocation to the Tensor class
// Long term this is closer to what we want.
// c.f. the fork of Kompute used in llama.cpp had 'Allow pre-allocated memory from vulkan for staging host.'
// as one of the first changes made.
// https://github.com/KomputeProject/kompute/commit/d3ad3aa657a8732f23cca11f2286a1f9196b94c8
//
#ifdef VULKAN_EP_USE_VMA
VulkanBufferAllocator::VulkanBufferAllocator(OrtDevice device, VmaAllocator& allocator)
    : IAllocator(CreateMemoryInfo(device)),
      allocator_{allocator},
      allocate_device_memory_{device.MemType() == OrtDevice::MemType::DEFAULT} {
  ;
}

void* VulkanBufferAllocator::Alloc(size_t size) {
  auto* kompute_tensor = new KomputeTensor(allocator_, narrow<uint32_t>(size), allocate_device_memory_);

  // we return this as the data pointer. consumer needs to
  return kompute_tensor;
}

void VulkanBufferAllocator::Free(void* p) {
  KomputeTensor* tensor = static_cast<KomputeTensor*>(p);
  tensor->destroy();
  delete tensor;
}
#else
VulkanBufferAllocator::VulkanBufferAllocator(OrtDevice device, kp::Manager& manager)
    : IAllocator(CreateMemoryInfo(device)), manager_{manager} {
}

void* VulkanBufferAllocator::Alloc(size_t size) {
  // this creates staging/device memory and copies to staging

  // we need to set this as the staging or primary buffer in the kp::Tensor depending on usage.
  //

  static const std::vector<float> dummy = {1.2f, 3.4f, 5.6f, 7.8f};
  auto tensor = manager_.tensor(dummy);
}

void VulkanBufferAllocator::Free(void* p) {
  delete p;
}
#endif

}  // namespace vulkan
}  // namespace onnxruntime
