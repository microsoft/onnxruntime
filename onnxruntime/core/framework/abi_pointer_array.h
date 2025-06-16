// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/session/onnxruntime_c_api.h"

struct OrtConstPointerArray {
  OrtConstPointerArray() = default;
  explicit OrtConstPointerArray(OrtTypeTag elem_type) : elem_type(elem_type) {}
  OrtConstPointerArray(OrtTypeTag elem_type, size_t size, void* initial_val = nullptr)
      : elem_type(elem_type), storage(size, initial_val) {}

  void Resize(size_t size, void* initial_val = nullptr) {
    storage.resize(size, initial_val);
  }

  template <typename T>
  gsl::span<T* const> ToConstSpan() const {
    return gsl::span<T* const>(reinterpret_cast<T* const*>(storage.data()), storage.size());
  }

  template <typename T>
  gsl::span<T*> ToSpan() {
    return gsl::span<T*>(reinterpret_cast<T**>(storage.data()), storage.size());
  }

  OrtTypeTag elem_type = OrtTypeTag::ORT_TYPE_TAG_Void;
  std::vector<void*> storage;
};
