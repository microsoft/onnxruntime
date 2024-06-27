// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_allocator.h"

#include <iostream>

#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace vulkan {

VulkanBufferAllocator::VulkanBufferAllocator()
    : IAllocator(OrtMemoryInfo("VulkanAllocator",
                               OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, /*device_id*/ 0),
                               /*id*/ 0,
                               OrtMemTypeDefault)) {
}

void* VulkanBufferAllocator::Alloc(size_t /*size*/) {
  ORT_NOT_IMPLEMENTED();
}

void VulkanBufferAllocator::Free(void* /*p*/) {
  ORT_NOT_IMPLEMENTED();
}

}  // namespace vulkan
}  // namespace onnxruntime
