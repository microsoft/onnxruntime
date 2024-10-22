// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_cxx_api.h"

/**
 * A container for OrtValues for RAII
 */
class OrtValueArray {
 private:
  std::vector<OrtValue*> values;

 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtValueArray);
  // n must be non-negative
  OrtValueArray(int n) : values(static_cast<size_t>(n), nullptr){};
  ~OrtValueArray() {
    for (OrtValue* v : values) {
      if (v != nullptr) Ort::GetApi().ReleaseValue(v);
    }
  }
  OrtValue* Get(size_t index) {
    return values[index];
  }
  void Set(size_t index, OrtValue* p) {
    values[index] = p;
  }

  OrtValue** Data() {
    return values.data();
  }

  int Length() {
    return static_cast<int>(values.size());
  }
};
