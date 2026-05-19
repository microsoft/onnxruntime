// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <mutex>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/providers/tensorrt/nv_includes.h"
#include "core/framework/allocator.h"

namespace onnxruntime {

class TensorRTGpuAllocator final : public nvinfer1::IGpuAllocator {
 public:
  explicit TensorRTGpuAllocator(IAllocator* allocator) : allocator_{allocator} {
    ORT_ENFORCE(allocator_ != nullptr);
  }

  void* allocate(uint64_t const size, uint64_t const alignment,
                 nvinfer1::AllocatorFlags const /*flags*/) noexcept override {
    return Allocate(size, alignment);
  }

  bool deallocate(void* const memory) noexcept override {
    return Deallocate(memory);
  }

#if NV_TENSORRT_MAJOR >= 10
  void* allocateAsync(uint64_t const size, uint64_t const alignment,
                      nvinfer1::AllocatorFlags const /*flags*/, cudaStream_t /*stream*/) noexcept override {
    return Allocate(size, alignment);
  }

  bool deallocateAsync(void* const memory, cudaStream_t stream) noexcept override {
    if (stream != nullptr && cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }
    return Deallocate(memory);
  }
#endif

 private:
  static bool IsAligned(void* p, uint64_t alignment) noexcept {
    if (p == nullptr || alignment <= 1) {
      return true;
    }
    if (alignment > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
      return false;
    }

    return reinterpret_cast<uintptr_t>(p) % static_cast<uintptr_t>(alignment) == 0;
  }

  static void* AlignPointer(void* p, uint64_t alignment) noexcept {
    if (p == nullptr || alignment <= 1) {
      return p;
    }
    if (alignment > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
      return nullptr;
    }

    const auto address = reinterpret_cast<uintptr_t>(p);
    const auto alignment_value = static_cast<uintptr_t>(alignment);
    const auto misalignment = address % alignment_value;
    if (misalignment == 0) {
      return p;
    }

    const auto padding = alignment_value - misalignment;
    if (address > std::numeric_limits<uintptr_t>::max() - padding) {
      return nullptr;
    }

    return reinterpret_cast<void*>(address + padding);
  }

  void* Allocate(uint64_t size, uint64_t alignment) noexcept {
    if (size == 0 || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return nullptr;
    }

    try {
      void* raw = allocator_->Alloc(static_cast<size_t>(size));
      if (raw == nullptr) {
        return nullptr;
      }

      if (IsAligned(raw, alignment)) {
        std::lock_guard<std::mutex> lock(lock_);
        raw_allocations_[raw] = raw;
        return raw;
      }

      allocator_->Free(raw);

      const uint64_t alignment_padding = alignment > 1 ? alignment - 1 : 0;
      if (alignment_padding > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
          size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - alignment_padding) {
        return nullptr;
      }

      raw = allocator_->Alloc(static_cast<size_t>(size + alignment_padding));
      if (raw == nullptr) {
        return nullptr;
      }

      void* aligned = AlignPointer(raw, alignment);
      if (aligned == nullptr) {
        allocator_->Free(raw);
        return nullptr;
      }

      std::lock_guard<std::mutex> lock(lock_);
      raw_allocations_[aligned] = raw;
      return aligned;
    } catch (...) {
      return nullptr;
    }
  }

  bool Deallocate(void* memory) noexcept {
    if (memory == nullptr) {
      return true;
    }

    try {
      void* raw = nullptr;
      {
        std::lock_guard<std::mutex> lock(lock_);
        auto it = raw_allocations_.find(memory);
        if (it == raw_allocations_.end()) {
          return false;
        }

        raw = it->second;
        raw_allocations_.erase(it);
      }

      allocator_->Free(raw);
      return true;
    } catch (...) {
      return false;
    }
  }

  IAllocator* allocator_;
  std::mutex lock_;
  InlinedHashMap<void*, void*> raw_allocations_;
};

}  // namespace onnxruntime
