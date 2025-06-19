// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/session/onnxruntime_c_api.h"

struct OrtArrayOfConstObjects {
  OrtArrayOfConstObjects() = default;
  explicit OrtArrayOfConstObjects(OrtTypeTag object_type) : object_type(object_type) {}
  OrtArrayOfConstObjects(OrtTypeTag object_type, size_t size, const void* initial_val = nullptr)
      : object_type(object_type), storage(size, initial_val) {}

  OrtTypeTag object_type = OrtTypeTag::ORT_TYPE_TAG_Void;
  std::vector<const void*> storage;
};
