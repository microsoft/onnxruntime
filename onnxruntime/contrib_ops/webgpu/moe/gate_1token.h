// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class Gate1TokenProgram final : public Program<Gate1TokenProgram> {
 public:
  Gate1TokenProgram(int k, bool is_fp16, bool has_router_weights, bool normalize_routing_weights,
                    bool sort_experts = false, bool use_subgroups = false);

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32});

 private:
  int k_;
  bool is_fp16_;
  bool has_router_weights_;
  bool normalize_routing_weights_;
  bool sort_experts_;
  bool use_subgroups_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
