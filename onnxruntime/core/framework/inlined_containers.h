// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file contains code and comments derived from llvm/ADT/SmallVector.h
// 
// Specifically CalculateInlinedVectorDefaultInlinedElements<T>() template is derived from 
// CalculateSmallVectorDefaultInlinedElements<T>() and its comments.

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
/// Inspired by LLVM SmallVector with ONNX Runtime adjuments for abseil.
///
/// Helper class for calculating the default number of inline elements for
/// `InlinedVector<T>`.
/// This produces the following on MSVC x64
///    int8_t  -> 41
//     int16_t -> 21
//     int32_t -> 11 
//     int64_t -> 6
//     std::string 40 -> 1
template<typename T>
struct CalculateInlinedVectorDefaultInlinedElements {
  // Parameter controlling the default number of inlined elements
  // for `InlinedVector<T>`.
  //
  // The default number of inlined elements ensures that
  // 1. There is at least one inlined element.
  // 2. `sizeof(InlinedVector<T>) <= kPreferredInlinedVectorSizeof` unless
  // it contradicts 1.
  static constexpr size_t kPreferredInlinedVectorSizeof = 64;

  // static_assert that sizeof(T) is not "too big".
  //
  // Because the InlinedVector must have at least one inlined element, it is possible
  // for an arbitrarily large inlined element to allocate an arbitrarily large
  // amount of inline storage. So we want to call attention to these cases and
  // make sure that users are making an intentional decision if they request a lot of inline storage.
  //
  // We want this assertion to trigger in pathological cases, but otherwise
  // not be too easy to hit. To accomplish that, the cutoff is actually somewhat
  // larger than kPreferredInlinedVectorSizeof (otherwise,
  // `InlinedVector<InlinedVector<T>>` would be one easy way to trip it, and that
  // pattern seems useful in practice).
  //
  // One wrinkle is that this assertion is in theory non-portable, since
  // sizeof(absl::InlinedVector<T, 1>) is in general platform-dependent. However, we don't expect this
  // to be much of an issue, because most LLVM development happens on 64-bit
  // hosts, and therefore sizeof(T) is expected to *decrease* when compiled for
  // 32-bit hosts, dodging the issue. The reverse situation, where development
  // happens on a 32-bit host and then fails due to sizeof(T) *increasing* on a
  // 64-bit host, is expected to be very rare.
  static_assert(
      sizeof(absl::InlinedVector<T, 1>) <= kPreferredInlinedVectorSizeof,
      "You are trying to use a default number of inlined elements for "
      "`InlinedVector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `InlinedVector<T, N>` to make "
      "sure you really want that much inline storage.");

  // Discount the size of the header itself when calculating the maximum inline
  // bytes.
  static constexpr size_t PreferredInlineBytes =
      kPreferredInlinedVectorSizeof - (sizeof(absl::InlinedVector<T, 1>) - sizeof(T));
  static constexpr size_t NumElementsThatFit = PreferredInlineBytes / sizeof(T);
  static constexpr size_t value =
      NumElementsThatFit == 0 ? 1 : NumElementsThatFit;
};

// Use InlinedVector for small arrays that can fit on a stack with a default
// value pre-calculated.
// Use TensorShapeVector for shapes.
template <typename T, 
          size_t N = CalculateInlinedVectorDefaultInlinedElements<T>::value,
          typename Allocator = std::allocator<T>>
using InlinedVector = absl::InlinedVector<T, N, Allocator>;

// InlinedHashSet and InlinedHashMap are preferred
// hash based containers. They store their values in the
// buckets array that is allocated in one shot. It eliminates
// per-node new/delete calls. Always call reserve() on any hash set/map
// when the number of items is known in advance
template <typename T, 
          typename Hash = absl::container_internal::hash_default_hash<T>,
          typename Eq = absl::container_internal::hash_default_eq<T>,
          typename Allocator = std::allocator<T>>
using InlinedHashSet = absl::flat_hash_set<T, Hash, Eq, Allocator>;

template <typename K, typename V,
          typename Hash = absl::container_internal::hash_default_hash<K>,
          typename Eq = absl::container_internal::hash_default_eq<K>,
          typename Allocator = std::allocator<std::pair<const K, V>>>
using InlinedHashMap = absl::flat_hash_map<K, V, Hash, Eq, Allocator>;

}  // namespace onnxruntime
