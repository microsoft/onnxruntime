// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/bert/causal_conv_with_state.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// Program for VarlenCausalConvWithState. One shader invocation owns a single
// (request, channel) pair and walks every token of its request serially, so
// request boundaries and per-channel carry state stay isolated.
class VarlenCausalConvWithStateProgram final : public Program<VarlenCausalConvWithStateProgram> {
 public:
  VarlenCausalConvWithStateProgram(bool has_bias, bool has_state, bool state_in_final_state,
                                   bool has_state_update, bool has_capture_count, bool use_silu)
      : Program{"VarlenCausalConvWithState"},
        has_bias_(has_bias),
        has_state_(has_state),
        state_in_final_state_(state_in_final_state),
        has_state_update_(has_state_update),
        has_capture_count_(has_capture_count),
        use_silu_(use_silu) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"kernel_size", ProgramUniformVariableDataType::Uint32},
      {"dilation", ProgramUniformVariableDataType::Uint32},
      {"pad", ProgramUniformVariableDataType::Uint32},
      {"state_update_capacity", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"num_invocations", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool has_state_;
  bool state_in_final_state_;
  bool has_state_update_;
  bool has_capture_count_;
  bool use_silu_;
};

// Kernel for VarlenCausalConvWithState (packed token-major, variable-length batches).
class VarlenCausalConvWithState final : public WebGpuKernel {
 public:
  VarlenCausalConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  CausalConvActivation activation_;
  int dilation_;
  int state_update_capacity_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
