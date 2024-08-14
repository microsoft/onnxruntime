// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Only one .cc file can include the implementation details of Vulkan Memory Allocator.
// Due to that we use this header as the preferred way to include vk_mem_alloc.h
// as well as to expose helper functions to expose internals as needed.
//
// VulkanMemoryAllocator must be included this way, so use this header and do not include vk_mem_alloc.h directly
// #define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace vma {

const VkDevice GetAllocatorDevice(VmaAllocator allocator);

}  // namespace vma
