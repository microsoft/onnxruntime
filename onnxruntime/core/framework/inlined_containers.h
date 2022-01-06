// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <memory_resource>
#include <core/common/safeint.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif

#include <absl/container/inlined_vector.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

namespace inline_containers_internal {
// abseil specific code
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

}  // namespace onnxruntime
