// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

namespace onnxruntime {
namespace contrib {
namespace webgpu {
namespace engram_helper {

// WGSL snippets shared by the contrib kernels. Append them to ShaderHelper::AdditionalImplementation().

// Numerically stable logistic function.
constexpr std::string_view kStableSigmoidWgsl =
    "fn stable_sigmoid(x: f32) -> f32 {\n"
    "  if (x > 0.0) { return 1.0 / (1.0 + exp(-x)); }\n"
    "  let e = exp(x);\n"
    "  return e / (1.0 + e);\n"
    "}\n";

// Engram gate pre-activation: sign(dot) * sqrt(max(abs(dot), 1e-6)). WGSL sign() already maps zero
// to zero, so a zero dot product yields a zero argument (and therefore a gate of exactly 0.5).
constexpr std::string_view kEngramGateArgWgsl =
    "fn engram_gate_arg(dot_value: f32) -> f32 {\n"
    "  return sign(dot_value) * sqrt(max(abs(dot_value), 0.000001));\n"
    "}\n";

// Euclidean modulo: the result always has the sign of `mod_value`, which must be positive.
constexpr std::string_view kPositiveModWgsl =
    "fn positive_mod(value: i32, mod_value: i32) -> i32 {\n"
    "  var result = value % mod_value;\n"
    "  if (result < 0i) { result += mod_value; }\n"
    "  return result;\n"
    "}\n";

}  // namespace engram_helper
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
