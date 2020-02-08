// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {
// This is a version of std::optional with limited functionality and plenty of
// room to improve. We should use std::optional when we move to C++17.
template <typename T>
class optional {
 public:
  optional() : has_value_{false}, value_{} {}

  optional(const optional&) = default;
  optional& operator=(const optional&) = default;
  optional(optional&&) = default;
  optional& operator=(optional&&) = default;

  optional(T value) : has_value_{true}, value_{value} {}
  optional& operator=(T value) {
    has_value_ = true;
    value_ = value;
    return *this;
  }

  bool has_value() const { return has_value_; }
  const T& value() const {
    ORT_ENFORCE(has_value_);
    return value_;
  }
  T& value() {
    ORT_ENFORCE(has_value_);
    return value_;
  }

 private:
  bool has_value_;
  T value_;
};
}  // namespace onnxruntime
