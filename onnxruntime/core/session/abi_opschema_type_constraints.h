// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/session/onnxruntime_c_api.h"

/// Container holding precomputed type constraint information from an OrtOpSchema.
/// Each entry represents one type constraint (e.g., "T", "T1") and includes
/// the allowed data types and which input/output formal parameters use it.
struct OrtOpSchemaTypeConstraints {
  struct Entry {
    std::string type_param_str;                   // e.g., "T"
    std::vector<std::string> allowed_type_strs;   // e.g., {"tensor(float)", "tensor(double)"}
    std::vector<const char*> allowed_type_ptrs;   // C API view into allowed_type_strs
    std::vector<size_t> input_indices;            // input indices using this constraint
    std::vector<size_t> output_indices;           // output indices using this constraint
  };

  std::vector<Entry> entries;
};
