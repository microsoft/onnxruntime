// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

// Shared-memory mixed-radix (2/3/4/5) Stockham FFT: one transform per workgroup, O(N log N).
// Used when the transform length is 5-smooth and fits in workgroup memory.
#define WEBGPU_D_F_T_PROGRAM_CONFIG(F) \
  F(uint32_t, length_)                 \
  F(uint32_t, input_components_)       \
  F(uint32_t, output_components_)      \
  F(bool, is_inverse_)                 \
  F(bool, is_onesided_)

struct DFTProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_D_F_T_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t length, uint32_t input_components, uint32_t output_components, bool is_inverse, bool is_onesided) {
    Config result{};
    result.length_ = uint32_t{length};
    result.input_components_ = uint32_t{input_components};
    result.output_components_ = uint32_t{output_components};
    result.is_inverse_ = bool{is_inverse};
    result.is_onesided_ = bool{is_onesided};

    return result;
  }
  static constexpr std::string_view name = "DFT";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_D_F_T_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch)                              \
  U(Uint32, signal_length)                      \
  U(Uint32, inner)                              \
  U(Uint32, output_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_D_F_T_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_D_F_T_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_D_F_T_PROGRAM_CONFIG

using DFTProgram = ConfiguredProgram<DFTProgramShader>;

// Direct O(N^2) DFT for lengths the shared-memory FFT cannot take (non 5-smooth, or beyond the
// workgroup memory budget). One workgroup per transform; each output bin sums over the input samples.
#define WEBGPU_D_F_T_DIRECT_PROGRAM_CONFIG(F) \
  F(uint32_t, length_)                        \
  F(uint32_t, input_components_)              \
  F(uint32_t, output_components_)             \
  F(bool, is_inverse_)                        \
  F(bool, is_onesided_)

struct DFTDirectProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_D_F_T_DIRECT_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t length, uint32_t input_components, uint32_t output_components, bool is_inverse, bool is_onesided) {
    Config result{};
    result.length_ = uint32_t{length};
    result.input_components_ = uint32_t{input_components};
    result.output_components_ = uint32_t{output_components};
    result.is_inverse_ = bool{is_inverse};
    result.is_onesided_ = bool{is_onesided};

    return result;
  }
  static constexpr std::string_view name = "DFTDirect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_D_F_T_DIRECT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch)                                     \
  U(Uint32, signal_length)                             \
  U(Uint32, inner)                                     \
  U(Uint32, output_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_D_F_T_DIRECT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_D_F_T_DIRECT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_D_F_T_DIRECT_PROGRAM_CONFIG

using DFTDirectProgram = ConfiguredProgram<DFTDirectProgramShader>;

class DFT final : public WebGpuKernel {
 public:
  DFT(const OpKernelInfo& info) : WebGpuKernel{info} {
    opset_ = info.node().SinceVersion();
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 0) != 0;
    is_inverse_ = info.GetAttrOrDefault<int64_t>("inverse", 0) != 0;
    // Opset 20 moves axis from an attribute to input 2; -2 is its spec default.
    axis_ = opset_ < 20 ? info.GetAttrOrDefault<int64_t>("axis", 1) : -2;
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  bool is_onesided_;
  bool is_inverse_;
  int opset_;
};

}  // namespace webgpu
}  // namespace onnxruntime
