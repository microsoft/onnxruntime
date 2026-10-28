// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu {

inline void AppendNormalizationCenter(ShaderHelper& shader, int components) {
  shader.AdditionalImplementation() << R"(
// Keep the exact nearby-value subtraction separate from the small mean offset,
// even on shader compilers that reassociate floating-point additions.
fn norm_center_scalar(value: f32, shift: f32, mean: f32) -> f32 {
  if (shift == 0.0) { return value - mean; }
  let a = frexp(value);
  let b = frexp(shift);
  let exponent = min(a.exp, b.exp);
  let a_mantissa = i32(ldexp(a.fract, a.exp - exponent + 24));
  let b_mantissa = i32(ldexp(b.fract, b.exp - exponent + 24));
  return ldexp(f32(a_mantissa - b_mantissa) * 0x1p-24, exponent) - mean;
}
)";
  const std::string type = components == 1 ? "f32" : "vec" + std::to_string(components) + "<f32>";
  shader.AdditionalImplementation() << "fn norm_center(value: " << type << ", shift: " << type
                                    << ", mean: " << type << ") -> " << type << " { return " << type << "(";
  for (int i = 0; i < components; ++i) {
    const std::string index = components == 1 ? "" : "[" + std::to_string(i) + "]";
    if (i) shader.AdditionalImplementation() << ", ";
    shader.AdditionalImplementation() << "norm_center_scalar(value" << index << ", shift" << index << ", mean" << index << ")";
  }
  shader.AdditionalImplementation() << "); }\n";
}

// Emits norm_shift and norm_mean separately so normalization need not round their sum.
// load(index) returns one scalar/vec2/vec4 of input values for this workgroup's row.
template <typename Load>
void AppendStableNormalizationMean(ShaderHelper& shader, int components,
                                   const std::string& vector_count, const std::string& element_count,
                                   Load load) {
  AppendNormalizationCenter(shader, components);
  shader.AdditionalImplementation()
      << "alias norm_vec_t = " << (components == 4 ? "vec4<f32>" : components == 2 ? "vec2<f32>"
                                                                                   : "f32")
      << ";\n"
      // Integer mantissa differences prevent fast-math reassociation from discarding the correction.
      << R"(
fn norm_compensated_add(acc: vec2<f32>, value: f32) -> vec2<f32> {
  let sum = acc.x + value;
  let larger = select(value, acc.x, abs(acc.x) >= abs(value));
  let smaller = select(acc.x, value, abs(acc.x) >= abs(value));
  if (sum == larger) { return vec2<f32>(sum, acc.y + smaller); }
  var error = 0.0;
  if ((larger < 0.0) == (smaller < 0.0) || abs(smaller) < 0.5 * abs(larger)) {
    let a = frexp(larger);
    let b = frexp(sum);
    let exponent = min(a.exp, b.exp);
    let a_mantissa = i32(ldexp(a.fract, a.exp - exponent + 24));
    let b_mantissa = i32(ldexp(b.fract, b.exp - exponent + 24));
    error = ldexp(f32(a_mantissa - b_mantissa) * 0x1p-24, exponent) + smaller;
  }
  return vec2<f32>(sum, acc.y + error);
}
var<workgroup> norm_sum: array<vec2<f32>, workgroup_size_x>;
var<workgroup> norm_shifted_sum: array<vec2<f32>, workgroup_size_x>;
var<workgroup> norm_shiftable: array<u32, workgroup_size_x>;
)";
  auto& body = shader.MainFunctionBody();
  body << "let norm_first_value = norm_vec_t(" << load("0u") << ");\n"
       << "let norm_first = " << (components == 1 ? "norm_first_value" : "norm_first_value.x") << ";\n"
       << "var norm_acc = vec2<f32>(0); var norm_shifted_acc = vec2<f32>(0);\n"
       << "var norm_can_shift = abs(norm_first) >= 0x1p-126 && abs(norm_first) <= 0x1.fffffep127;\n"
       << "for (var norm_i = local_idx; norm_i < " << vector_count << "; norm_i += workgroup_size_x) {\n"
       << "  let norm_value = norm_vec_t(" << load("norm_i") << ");\n";
  for (int component = 0; component < components; ++component) {
    const std::string value = components == 1 ? "norm_value" : "norm_value[" + std::to_string(component) + "]";
    body << "  norm_acc = norm_compensated_add(norm_acc, " << value << ");\n"
         // Sterbenz's lemma makes these same-sign, nearby subtractions exact.
         << "  norm_can_shift = norm_can_shift && ((" << value << " < 0.0) == (norm_first < 0.0)) &&\n"
         << "                   abs(" << value << ") >= 0.5 * abs(norm_first) && 0.5 * abs(" << value << ") <= abs(norm_first);\n"
         << "  if (norm_can_shift) { norm_shifted_acc = norm_compensated_add(norm_shifted_acc, " << value << " - norm_first); }\n";
  }
  body << "}\n"
       << "norm_sum[local_idx] = norm_acc; norm_shifted_sum[local_idx] = norm_shifted_acc;\n"
       << "norm_shiftable[local_idx] = u32(norm_can_shift);\n"
       << "workgroupBarrier();\n"
       << "var norm_width = workgroup_size_x;\n"
       << "for (var norm_half = norm_width >> 1u; norm_half > 0u; norm_half = norm_width >> 1u) {\n"
       << "  norm_width = norm_half + (norm_width & 1u);\n"
       << "  if (local_idx < norm_half) {\n"
       << "    let other = norm_sum[local_idx + norm_width];\n"
       << "    norm_sum[local_idx] = norm_compensated_add(norm_compensated_add(norm_sum[local_idx], other.x), other.y);\n"
       << "    let shifted = norm_shifted_sum[local_idx + norm_width];\n"
       << "    norm_shifted_sum[local_idx] = norm_compensated_add(norm_compensated_add(norm_shifted_sum[local_idx], shifted.x), shifted.y);\n"
       << "    norm_shiftable[local_idx] &= norm_shiftable[local_idx + norm_width];\n"
       << "  }\n  workgroupBarrier();\n}\n"
       << "let norm_use_shift = norm_shiftable[0] != 0u;\n"
       << "let norm_shift = select(0.0, norm_first, norm_use_shift);\n"
       << "let norm_total = select(norm_sum[0], norm_shifted_sum[0], norm_use_shift);\n"
       << "let norm_mean_parts = norm_compensated_add(vec2<f32>(norm_total.x / f32(" << element_count << "), 0),\n"
       << "                                            norm_total.y / f32(" << element_count << "));\n"
       << "let norm_mean = norm_mean_parts.x + norm_mean_parts.y;\n";
}

}  // namespace onnxruntime::webgpu
