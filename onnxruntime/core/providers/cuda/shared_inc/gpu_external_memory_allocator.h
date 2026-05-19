// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <mutex>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"

namespace onnxruntime {

class GpuExternalMemoryAllocator : public IAllocator {
 public:
  GpuExternalMemoryAllocator(OrtDevice::DeviceId device_id, const char* name, void* mem_ptr, size_t mem_size)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                                    device_id),
                          OrtMemTypeDefault)),
        base_{reinterpret_cast<uintptr_t>(mem_ptr)},
        size_{mem_size} {
    ORT_ENFORCE(mem_ptr != nullptr && mem_size > 0);
    ORT_ENFORCE(base_ <= std::numeric_limits<uintptr_t>::max() - mem_size);
    free_blocks_.push_back({0, mem_size});
  }

  void* Alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(lock_);
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
      const auto aligned_offset = AlignOffset(it->offset);
      const auto padding = aligned_offset - it->offset;
      if (it->size < padding || it->size - padding < size) {
        continue;
      }

      const auto allocated_block = Block{aligned_offset, size};
      const auto suffix_offset = static_cast<size_t>(SafeInt<size_t>(aligned_offset) + size);
      const auto block_end = static_cast<size_t>(SafeInt<size_t>(it->offset) + it->size);
      const auto suffix_size = block_end - suffix_offset;
      const auto prefix_size = padding;

      if (prefix_size > 0 && suffix_size > 0) {
        it->size = prefix_size;
        free_blocks_.insert(it + 1, {suffix_offset, suffix_size});
      } else if (prefix_size > 0) {
        it->size = prefix_size;
      } else if (suffix_size > 0) {
        it->offset = suffix_offset;
        it->size = suffix_size;
      } else {
        free_blocks_.erase(it);
      }

      void* p = reinterpret_cast<void*>(base_ + allocated_block.offset);
      allocated_blocks_[p] = allocated_block;
      return p;
    }

    ORT_THROW("External GPU memory buffer exhausted. Requested ", size, " bytes from a ", size_, " byte buffer.");
  }

  void Free(void* p) override {
    if (p == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(lock_);
    auto it = allocated_blocks_.find(p);
    ORT_ENFORCE(it != allocated_blocks_.end());
    const auto block = it->second;
    allocated_blocks_.erase(it);
    InsertFreeBlock(block);
  }

  void* Reserve(size_t size) override {
    return Alloc(size);
  }

 private:
  struct Block {
    size_t offset;
    size_t size;
  };

  size_t AlignOffset(size_t offset) const {
    const auto address = base_ + offset;
    const auto misalignment = address % kMinAlignment;
    if (misalignment == 0) {
      return offset;
    }

    const auto padding = kMinAlignment - misalignment;
    return static_cast<size_t>(SafeInt<size_t>(offset) + padding);
  }

  void InsertFreeBlock(Block block) {
    auto insert_it = free_blocks_.begin();
    while (insert_it != free_blocks_.end() && insert_it->offset < block.offset) {
      ++insert_it;
    }
    auto inserted_it = free_blocks_.insert(insert_it, block);

    if (inserted_it != free_blocks_.begin()) {
      auto previous_it = inserted_it - 1;
      const auto previous_end = static_cast<size_t>(SafeInt<size_t>(previous_it->offset) + previous_it->size);
      if (previous_end == inserted_it->offset) {
        previous_it->size = static_cast<size_t>(SafeInt<size_t>(previous_it->size) + inserted_it->size);
        inserted_it = free_blocks_.erase(inserted_it);
        inserted_it = previous_it;
      }
    }

    auto next_it = inserted_it + 1;
    const auto inserted_end = static_cast<size_t>(SafeInt<size_t>(inserted_it->offset) + inserted_it->size);
    if (next_it != free_blocks_.end() && inserted_end == next_it->offset) {
      inserted_it->size = static_cast<size_t>(SafeInt<size_t>(inserted_it->size) + next_it->size);
      free_blocks_.erase(next_it);
    }
  }

  static constexpr size_t kMinAlignment = 256;

  std::mutex lock_;
  uintptr_t base_;
  size_t size_;
  InlinedVector<Block> free_blocks_;
  InlinedHashMap<void*, Block> allocated_blocks_;
};

}  // namespace onnxruntime
