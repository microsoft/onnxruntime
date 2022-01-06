// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <iostream>
#include <memory_resource>
#include "core/platform/env.h"
#include "core/framework/allocator.h"

namespace onnxruntime {

// An STL wrapper for ORT allocators. This enables overriding the
// std::allocator used in STL containers for better memory performance.
template <class T>
class OrtStlAllocator {
  template <class U>
  friend class OrtStlAllocator;
  AllocatorPtr allocator_;

 public:
  typedef T value_type;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::true_type;

  OrtStlAllocator(const AllocatorPtr& a) noexcept {
    allocator_ = a;
  }
  OrtStlAllocator(const OrtStlAllocator& other) noexcept {
    allocator_ = other.allocator_;
  }
  template <class U>
  OrtStlAllocator(const OrtStlAllocator<U>& other) noexcept {
    allocator_ = other.allocator_;
  }

  T* allocate(size_t n, const void* hint = 0) {
    ORT_UNUSED_PARAMETER(hint);
    return reinterpret_cast<T*>(allocator_->Alloc(n * sizeof(T)));
  }

  void deallocate(T* p, size_t n) {
    ORT_UNUSED_PARAMETER(n);
    allocator_->Free(p);
  }
};

template <class T1, class T2>
bool operator==(const OrtStlAllocator<T1>& lhs, const OrtStlAllocator<T2>& rhs) noexcept {
  return lhs.allocator_ == rhs.allocator_;
}
template <class T1, class T2>
bool operator!=(const OrtStlAllocator<T1>& lhs, const OrtStlAllocator<T2>& rhs) noexcept {
  return lhs.allocator_ != rhs.allocator_;
}

namespace ort_stl_allocator_internal {
inline void* Allocate(size_t size, std::unique_ptr<uint8_t[]>& buf) {
  constexpr auto alignment = sizeof(std::max_align_t);
  size += alignment;
  buf = std::make_unique<uint8_t[]>(size);
  void* ptr = buf.get();
  return std::align(alignment, size, ptr, size);
}
}  // namespace ort_stl_allocator_internal

// This macro can serve to pre-allocate estimated memory size for a container.
// The allocation would take place on a stack up to a certain limit. With proper calculation
// one can utilize the stack for small maps/sets for both pmr::Inlined and std::pmr::map/unordered_map
// std::pmr::set/onordered_set. For std containers, this can reduce to 1 or even may completely eliminate
// the number of new/delete calls.
#define OrtDeclareAlignedStackOrAllocatedBuffer(buffer_ptr, size_in_bytes)                           \
  std::unique_ptr<uint8_t[]> on_heap_##buffer_ptr;                                                   \
  void* buffer_ptr = (IsSizeOverStackAllocationLimit(size_in_bytes))                                 \
                         ? ort_stl_allocator_internal::Allocate(size_in_bytes, on_heap_##buffer_ptr) \
                         : ORT_ALLOCA(size_in_bytes)

namespace pmr {
/// <summary>
/// This class provides a thin abstraction over the std::pmr::monotonic_buffer_resource
/// </summary>
class SmallBufferResource : public std::pmr::monotonic_buffer_resource {
 public:
  SmallBufferResource(void* ptr, size_t size_in_bytes)
      : monotonic_buffer_resource(ptr, size_in_bytes, std::pmr::get_default_resource()) {}
  SmallBufferResource(void* ptr, size_t size_in_bytes, std::pmr::memory_resource* upstream)
      : monotonic_buffer_resource(ptr, size_in_bytes, upstream) {}
};

// For development and debugging. Use this for logging memory
// allocations before the pre-allocated buffer and after. This allows
// to evaluate the efficiency of the allocation size estimates tracking
// allocation requests that are satisfied from the pre-allocated buffer
// and the ones that leak into the default allocator on the heap.
class DebugMemoryResource : public std::pmr::memory_resource {
 public:
  DebugMemoryResource(std::string name, std::pmr::memory_resource* up)
      : name_(std::move(name)), upstream_(up) {}

  size_t Allocated() const noexcept { return total_allocated_; }
  size_t Deallocated() const noexcept { return total_deallocated_; }

 private:
  std::string name_;
  std::pmr::memory_resource* upstream_;
  size_t total_allocated_ = 0;
  size_t total_deallocated_ = 0;

 private:
  void* do_allocate(size_t bytes, size_t align) override {
    std::cout << name_ << " : allocate : " << bytes << std::endl;
    total_allocated_ += bytes;
    return upstream_->allocate(bytes, align);
  }
  void do_deallocate(void* ptr, size_t bytes, size_t align) override {
    std::cout << name_ << " : deallocate : " << bytes << std::endl;
    total_deallocated_ += bytes;
    upstream_->deallocate(ptr, bytes, align);
  }
  bool do_is_equal(const memory_resource&) const noexcept override { return false; }
};
}  // namespace pmr

}  // namespace onnxruntime