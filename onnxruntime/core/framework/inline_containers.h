// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory_resource>
#include <absl/container/inlined_vector.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>
#include <core/framework/tensor_shape.h>

namespace onnxruntime {

template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

template <typename T>
using InlineHashSet = absl::flat_hash_set<T>;

template <typename K, typename V>
using InlineHashMap = absl::flat_hash_map<K, V>;

namespace pmr {
template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N, std::pmr::polymorphic_allocator<T>>;

template <typename T>
using InlineHashSet = absl::flat_hash_set<T, absl::container_internal::hash_default_hash<T>,
                                          absl::container_internal::hash_default_eq<T>,
                                          std::pmr::polymorphic_allocator<T>>;

template <typename K, typename V>
using InlineHashMap = absl::flat_hash_map<K, V, absl::container_internal::hash_default_hash<K>,
                                          absl::container_internal::hash_default_eq<K>,
                                          std::pmr::polymorphic_allocator<std::pair<const K, V>>>;
}  // namespace pmr

}  // namespace onnxruntime
