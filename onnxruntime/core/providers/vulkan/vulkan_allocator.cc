// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_allocator.h"

#include <iostream>

#include "ncnn-src/src/allocator.h"

#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace vulkan {

// Start simple - direct usage of NCNN allocators with no arena
//
// TODO: Figure out how we want to manage allocators and caching
// Do we want to use BFCArena?
// Do we need/want to split by usage as there are different Vulkan flags for the buffers
//
// e.g.
//   for initializers it might be optimal to figure out total size and do one allocation that is used read-only
//   for execution data we want larger backing allocations that the buffers use (bind buffer to memory at offset).
//
// NCNN VkBlobAllocator implementation looks like it is simplistic and does allocation + buffer per fastMalloc.
// We might really want a large allocation in the background with fastMalloc binding a buffer at an offset.
//
// We may also need logic to pick from multiple potential heaps with different Vulkan flags.
//   see https://www.gdcvault.com/play/1025458/Advanced-Graphics-Techniques-Tutorial-New for some slides showing
//   some different heaps/flags/sizes.
//
// We also might want to consider using the VulkanMemoryAllocator library to do optimal things more easily.
// Fortunately the NCNN Vulkan layers seem to be fairly nicely abstracted for this so it is a case of wiring in VMA
// to Mat/VkMat/Option classes as the allocator to used tends to be accessed via those.

namespace {
OrtMemoryInfo CreateMemoryInfo(OrtDevice device) {
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

VulkanBufferAllocator::VulkanBufferAllocator(OrtDevice device, ncnn::VkAllocator& allocator)
    : IAllocator(CreateMemoryInfo(device)), allocator_{allocator} {
}

void* VulkanBufferAllocator::Alloc(size_t size) {
  // pad to match NCNN behavior
  size_t aligned_size = ncnn::alignSize(size + NCNN_MALLOC_OVERREAD, 16);
  ncnn::VkBufferMemory* mem = allocator_.fastMalloc(aligned_size);

  return mem;
}

void VulkanBufferAllocator::Free(void* p) {
  return allocator_.fastFree(static_cast<ncnn::VkBufferMemory*>(p));
}

}  // namespace vulkan
}  // namespace onnxruntime
