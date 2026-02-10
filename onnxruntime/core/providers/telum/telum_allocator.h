// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "telum_common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Allocator for Telum EP that ensures 4K alignment required by zDNN
 *
 * zDNN requires all tensor buffers to be aligned on 4K boundaries for
 * optimal hardware performance. This allocator ensures proper alignment
 * and manages memory lifecycle for zDNN operations.
 */
class TelumAllocator : public IAllocator {
 public:
  TelumAllocator()
      : IAllocator(OrtMemoryInfo(TELUM, OrtAllocatorType::OrtDeviceAllocator)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  /**
   * @brief Allocate aligned memory
   * @param size Size in bytes to allocate
   * @param alignment Alignment requirement (must be power of 2)
   * @return Pointer to aligned memory or nullptr on failure
   */
  void* AllocateAligned(size_t size, size_t alignment);

  /**
   * @brief Free aligned memory
   * @param p Pointer to memory allocated by AllocateAligned
   */
  void FreeAligned(void* p);
};

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
