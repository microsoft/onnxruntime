// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <memory>
#include <iostream>
#include <memory_resource>
#include <core/common/safeint.h>

#include "core/framework/allocator.h"

#pragma warning(push)
#pragma warning(disable : 4127)
#include <absl/container/inlined_vector.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>
#pragma warning(pop)

namespace onnxruntime {

// Use InlinedVector for small arrays that can fit on a stack.
// Use TensorShapeVector for shapes.
template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

// InlinedHashSet and InlinedHashMap are preferred
// hash based containers. They store their values in the
// buckets array that is allocated in one shot. It eliminated
// per-node new/delete calls. Proper memory estimates combined with
// OrtDeclareAllignedStackOrAllocatedBuffer may reduce the number of needed
// allocated to 1 or completely place it on a stack.
template <typename T>
using InlinedHashSet = absl::flat_hash_set<T>;

template <typename K, typename V>
using InlinedHashMap = absl::flat_hash_map<K, V>;

namespace pmr {
template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N, std::pmr::polymorphic_allocator<T>>;

template <typename T, typename Hash = absl::container_internal::hash_default_hash<T>, typename Eq = absl::container_internal::hash_default_eq<T>>
using InlinedHashSet = absl::flat_hash_set<T, Hash, Eq, std::pmr::polymorphic_allocator<T>>;

template <typename K, typename V,
          typename Hash = absl::container_internal::hash_default_hash<K>,
          typename Eq = absl::container_internal::hash_default_eq<K>>
using InlinedHashMap = absl::flat_hash_map<K, V, Hash, Eq, std::pmr::polymorphic_allocator<std::pair<const K, V>>>;
}  // namespace pmr

// Default stack size on Windows - 320K(32-bit) or 1MB (64-bit), Linux - 8 MB
// MacOS - main - 1MB and other threads 512Kb
#ifdef _MSC_VER
#define ORT_ALLOCA(s) _alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#elif defined(__GNUC__) || defined(__clang__)
#define ORT_ALLOCA(s) alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#else
// always on the heap
#define ORT_ALLOCA(s) nullptr
constexpr size_t kOrtStackAllocationLimitBytes = 0;
#endif

namespace inline_containers_internal {
inline void* allocate(size_t size, std::unique_ptr<uint8_t[]>& buf) {
  constexpr auto alignment = sizeof(std::max_align_t);
  size += alignment;
  buf = std::make_unique<uint8_t[]>(size);
  void* ptr = buf.get();
  return std::align(alignment, size, ptr, size);
}

// absel specific code
inline size_t EstimateHashStorageSize(size_t slot_size, size_t num_elements) {
  // See https://abseil.io/docs/cpp/guides/container#memory-usage
  // However, the picture is a lot more complex
  // up to a power of two - 1 with minimum of 1
  constexpr size_t num_cloned_bytes = 15;

  const SafeInt<size_t> nelem = num_elements ? ~size_t{} >> absl::countl_zero(num_elements) : 1;
  const SafeInt<size_t> num_control_bytes = nelem + 1 + num_cloned_bytes;
  const SafeInt<size_t> slot_offset = (num_control_bytes + slot_size - 1) & (~slot_size + 1);
  return (slot_offset + nelem * slot_size);
}

}  // namespace inline_containers_internal

/// <summary>
/// Estimate memory requirements for an InlinedHashSet
/// so it can be pre-allocated on a stack or using other allocator when the number
/// of elements is known. This provides an oppty to bring the number of allocations
/// down to zero.
/// The InlinedHashSet keeps values in the buckets array which is allocated in one shot.
/// </summary>
/// <param name="num_elements">number of elements</param>
/// <returns></returns>
template <class T>
inline size_t EstimateInlinedHashSetMemory(size_t num_elements) {
  constexpr size_t slot_size = sizeof(InlinedHashSet<T>::slot_type);
  return inline_containers_internal::EstimateHashStorageSize(slot_size, num_elements);
}

/// <summary>
/// Estimate memory requirements for an InlinedHashMap
/// so it can be pre-allocated on a stack or using other allocator when the number
/// of elements is known. This provides an oppty to bring the number of allocations
/// down to zero.
/// The InlinedHashMap keeps values in the buckets array which is allocated in one shot.
/// </summary>
/// <param name="num_elements">number of elements</param>
/// <returns></returns>
template <class K, class V>
inline size_t EstimateInlinedHashMapMemory(size_t num_elements) {
  constexpr size_t slot_size = sizeof(InlinedHashMap<K, V>::slot_type);
  return inline_containers_internal::EstimateHashStorageSize(slot_size, num_elements);
}

inline bool IsSizeOverStackAllocationLimit(size_t size) {
  return size > kOrtStackAllocationLimitBytes;
}

// This macro can serve to pre-allocate estimated memory size for a container.
// The allocation would take place on a stack up to a certain limit. With proper calculation
// one can utilize the stack for small maps/sets for both pmr::Inlined and std::pmr::map/unordered_map
// std::pmr::set/onordered_set. For std containers, this can reduce to 1 or even may completely eliminate
// the number of new/delete calls.
#define OrtDeclareAllignedStackOrAllocatedBuffer(buffer_ptr, size_in_bytes)                          \
  std::unique_ptr<uint8_t[]> on_heap_##buffer_ptr;                                                   \
  void* buffer_ptr = (size_in_bytes > kOrtStackAllocationLimitBytes)                                 \
                         ? inline_containers_internal::allocate(size_in_bytes, on_heap_##buffer_ptr) \
                         : ORT_ALLOCA(size_in_bytes)

// This gives a set size stackbuffer
template <typename T, size_t N>
class SmallBuffer {
  T buffer_[N];

 public:
  T* Buffer() noexcept { return buffer_; }
  constexpr size_t size() const noexcept { return N; }
  constexpr size_t size_in_bytes() const noexcept { return sizeof(T) * N; }
};

class SmallBufferResource {
  std::pmr::monotonic_buffer_resource resource_;

 public:
  SmallBufferResource(void* ptr, size_t size_in_bytes)
      : resource_(ptr, size_in_bytes, std::pmr::get_default_resource()) {}
  SmallBufferResource(void* ptr, size_t size_in_bytes, std::pmr::memory_resource* upstream)
      : resource_(ptr, size_in_bytes, upstream) {}
  std::pmr::memory_resource* resource() noexcept { return &resource_; }
  std::pmr::memory_resource* upstream() const noexcept { return resource_.upstream_resource(); }
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

}  // namespace onnxruntime
