// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Include full implementation in this .cc file only
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace vma {
const VkDevice GetAllocatorDevice(VmaAllocator allocator) {
  return allocator->m_hDevice;
}
}  // namespace vma
