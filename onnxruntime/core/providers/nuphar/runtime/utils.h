// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/data_types.h"

// Common utilities for runtime
// Use inline if utilities are in a runtime critical path

namespace onnxruntime {
namespace nuphar {

inline void ShapeRemoveAxis(
    std::vector<int64_t>& shape,
    const int64_t* original_shape,
    size_t rank,
    size_t axis) {
  for (size_t i = 0; i < rank; ++i) {
    if (i != axis) {
      shape.push_back(original_shape[i]);
    }
  }
}

inline void ShapeInsertAxis(
    std::vector<int64_t>& shape,
    const int64_t* original_shape,
    size_t rank,
    size_t axis,
    int64_t value) {
  for (size_t i = 0; i < axis; ++i) {
    shape.push_back(original_shape[i]);
  }

  shape.push_back(value);

  for (size_t i = axis; i < rank; ++i) {
    shape.push_back(original_shape[i]);
  }
}

inline int64_t BytesOfShape(const std::vector<int64_t>& shape, MLDataType dtype) {
  int64_t total_size = dtype->Size();
  for (const auto& s : shape) {
    total_size *= s;
  }
  return total_size;
}

inline int64_t BytesOfShape(const int64_t* shape, size_t rank, MLDataType dtype) {
  int64_t total_size = dtype->Size();
  for (size_t i = 0; i < rank; ++i) {
    total_size *= shape[i];
  }
  return total_size;
}

#define STATIC_PROMOTE_FROM_BASE(X, BASE, KEY, VALUE) \
  static inline const X* Promote(const BASE* base) {  \
    auto derived = static_cast<const X*>(base);       \
    ORT_ENFORCE(nullptr != derived);                  \
    return derived;                                   \
  }                                                   \
                                                      \
  static inline X* Promote(BASE* base) {              \
    auto derived = dynamic_cast<X*>(base);            \
    ORT_ENFORCE(nullptr != derived);                  \
    return derived;                                   \
  }

}  // namespace nuphar
}  // namespace onnxruntime
