// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>

#ifdef _MSC_VER
#pragma warning(push)
// C4127: conditional expression is constant
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
// buckets array that is allocated in one shot. It eliminates
// per-node new/delete calls. Always call reserve() on any set/map
// be it a std container or not.
template <typename T>
using InlinedHashSet = absl::flat_hash_set<T>;

template <typename K, typename V>
using InlinedHashMap = absl::flat_hash_map<K, V>;

}  // namespace onnxruntime
