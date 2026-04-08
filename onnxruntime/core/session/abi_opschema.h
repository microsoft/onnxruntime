// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/session/onnxruntime_c_api.h"

namespace ONNX_NAMESPACE {
class OpSchema;
}  // namespace ONNX_NAMESPACE

/// A single type constraint entry (e.g., "T" or "T1") from an operator schema.
/// Holds the constraint name, allowed data types, and which input/output formal parameters use it.
/// Non-owning — lifetime is tied to the parent OrtOpSchema.
struct OrtOpSchemaTypeConstraint {
  std::string type_param_str;                  // e.g., "T"
  std::vector<std::string> allowed_type_strs;  // e.g., {"tensor(float)", "tensor(double)"}
  std::vector<const char*> allowed_type_ptrs;  // C API view into allowed_type_strs
  std::vector<size_t> input_indices;           // input indices using this constraint
  std::vector<size_t> output_indices;          // output indices using this constraint
};

/// Opaque struct wrapping an ONNX operator schema pointer and its precomputed type constraints.
/// Allocated by GetOpSchema and released by ReleaseOpSchema.
struct OrtOpSchema {
  const ONNX_NAMESPACE::OpSchema* onnx_schema;
  std::vector<OrtOpSchemaTypeConstraint> constraints;
  // O(1) lookup: input/output index → constraint pointer (nullptr if no constraint)
  std::vector<const OrtOpSchemaTypeConstraint*> input_to_constraint;
  std::vector<const OrtOpSchemaTypeConstraint*> output_to_constraint;
};
