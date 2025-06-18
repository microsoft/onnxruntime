// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/session/onnxruntime_c_api.h"

struct OrtArrayOfConstObjects {
  OrtArrayOfConstObjects() = default;
  explicit OrtArrayOfConstObjects(OrtTypeTag elem_type) : elem_type(elem_type) {}
  OrtArrayOfConstObjects(OrtTypeTag elem_type, size_t size, const void* initial_val = nullptr)
      : elem_type(elem_type), storage(size, initial_val) {}

  OrtTypeTag elem_type = OrtTypeTag::ORT_TYPE_TAG_Void;
  std::vector<const void*> storage;
};
