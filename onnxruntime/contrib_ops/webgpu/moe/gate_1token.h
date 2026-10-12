// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_GATE1_TOKEN_PROGRAM_CONFIG(F) \
  F(int, k_)                                 \
  F(bool, is_fp16_)                          \
  F(bool, has_router_weights_)               \
  F(bool, normalize_routing_weights_)

struct Gate1TokenProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATE1_TOKEN_PROGRAM_CONFIG);
  static Config MakeConfig(int k, bool is_fp16, bool has_router_weights, bool normalize_routing_weights);
  static constexpr std::string_view name = "QmoeGate1Token";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_GATE1_TOKEN_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, rows)                                     \
  U(Uint32, cols)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATE1_TOKEN_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATE1_TOKEN_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_GATE1_TOKEN_PROGRAM_CONFIG

using Gate1TokenProgram = ConfiguredProgram<Gate1TokenProgramShader>;

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
