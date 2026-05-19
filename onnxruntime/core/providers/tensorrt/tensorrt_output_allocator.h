// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/common/common.h"
#include "core/providers/tensorrt/nv_includes.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// Class to allocate memory for outputs with data-dependent shapes. The sizes of those are unknown so pre-allocation is
// not possible.
class OutputAllocator : public nvinfer1::IOutputAllocator {
 public:
  explicit OutputAllocator(OrtAllocator* allocator);

#if NV_TENSORRT_MAJOR >= 10
  void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
                              cudaStream_t stream) noexcept override;
#else
  void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override;
#endif
  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

  void* getBuffer() {
    return outputPtr;
  }

  std::vector<int64_t>& getOutputShape() {
    return output_shapes;
  }

  uint64_t getSize() {
    return allocated_size;
  }

  ~OutputAllocator() override {
    ReleaseBuffer();
  }

 private:
  void ReleaseBuffer() noexcept;
  void* Allocate(uint64_t size, uint64_t alignment) noexcept;
  static bool IsAligned(void* p, uint64_t alignment) noexcept;
  static void* AlignPointer(void* p, uint64_t alignment) noexcept;

  OrtAllocator* allocator_{nullptr};
  void* raw_output_ptr{nullptr};
  void* outputPtr{nullptr};
  uint64_t allocated_size = 0;
  std::vector<int64_t> output_shapes;
};

inline OutputAllocator::OutputAllocator(OrtAllocator* allocator) : allocator_{allocator} {
  ORT_ENFORCE(allocator_ != nullptr);
}

inline bool OutputAllocator::IsAligned(void* p, uint64_t alignment) noexcept {
  if (p == nullptr || alignment <= 1) {
    return true;
  }
  if (alignment > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
    return false;
  }

  return reinterpret_cast<uintptr_t>(p) % static_cast<uintptr_t>(alignment) == 0;
}

inline void* OutputAllocator::AlignPointer(void* p, uint64_t alignment) noexcept {
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

inline void OutputAllocator::ReleaseBuffer() noexcept {
  if (raw_output_ptr == nullptr) {
    return;
  }

  try {
    allocator_->Free(allocator_, raw_output_ptr);
  } catch (...) {
  }
  raw_output_ptr = nullptr;
  outputPtr = nullptr;
  allocated_size = 0;
}

inline void* OutputAllocator::Allocate(uint64_t size, uint64_t alignment) noexcept {
  if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return nullptr;
  }

  const auto requested_size = static_cast<size_t>(size);
  try {
    void* raw = allocator_->Alloc(allocator_, requested_size);
    if (raw == nullptr) {
      return nullptr;
    }

    if (IsAligned(raw, alignment)) {
      raw_output_ptr = raw;
      outputPtr = raw;
      allocated_size = size;
      return outputPtr;
    }

    allocator_->Free(allocator_, raw);

    const uint64_t alignment_padding = alignment > 1 ? alignment - 1 : 0;
    if (alignment_padding > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - alignment_padding) {
      return nullptr;
    }

    raw = allocator_->Alloc(allocator_, static_cast<size_t>(size + alignment_padding));
    if (raw == nullptr) {
      return nullptr;
    }

    void* aligned = AlignPointer(raw, alignment);
    if (aligned == nullptr) {
      allocator_->Free(allocator_, raw);
      return nullptr;
    }

    raw_output_ptr = raw;
    outputPtr = aligned;
    allocated_size = size;
    return outputPtr;
  } catch (...) {
    return nullptr;
  }
}

#if NV_TENSORRT_MAJOR >= 10
inline void* OutputAllocator::reallocateOutputAsync(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                                    uint64_t alignment, cudaStream_t /*stream*/) noexcept {
  // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
  // even for empty tensors, so allocate a dummy byte.
  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size || !IsAligned(outputPtr, alignment)) {
    ReleaseBuffer();
    outputPtr = Allocate(size, alignment);
  }
  // if allocation fails, returns nullptr.
  return outputPtr;
}
#else
// Only override this method when TensorRT <= 8.6
inline void* OutputAllocator::reallocateOutput(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                               uint64_t alignment) noexcept {
  // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
  // even for empty tensors, so allocate a dummy byte.
  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size || !IsAligned(outputPtr, alignment)) {
    ReleaseBuffer();
    outputPtr = Allocate(size, alignment);
  }
  // if allocation fails, returns nullptr.
  return outputPtr;
}
#endif

inline void OutputAllocator::notifyShape(char const* /*tensorName*/, nvinfer1::Dims const& dims) noexcept {
  output_shapes.clear();
  output_shapes.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; i++) {
    output_shapes.push_back(dims.d[i]);
  }
}

}  // namespace onnxruntime
