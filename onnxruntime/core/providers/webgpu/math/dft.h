// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

// Shared-memory mixed-radix (2/3/4/5) Stockham FFT: one transform per workgroup, O(N log N).
// Used when the transform length is 5-smooth and fits in workgroup memory.
class DFTProgram final : public Program<DFTProgram> {
 public:
  DFTProgram(uint32_t length, uint32_t input_components, uint32_t output_components, bool is_inverse, bool is_onesided)
      : Program{"DFT"},
        length_{length},
        input_components_{input_components},
        output_components_{output_components},
        is_inverse_{is_inverse},
        is_onesided_{is_onesided} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch", ProgramUniformVariableDataType::Uint32},
      {"signal_length", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"output_length", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t length_;
  uint32_t input_components_;
  uint32_t output_components_;
  bool is_inverse_;
  bool is_onesided_;
};

// Direct O(N^2) DFT for lengths the shared-memory FFT cannot take (non 5-smooth, or beyond the
// workgroup memory budget). One workgroup per transform; each output bin sums over the input samples.
class DFTDirectProgram final : public Program<DFTDirectProgram> {
 public:
  DFTDirectProgram(uint32_t length, uint32_t input_components, uint32_t output_components, bool is_inverse, bool is_onesided)
      : Program{"DFTDirect"},
        length_{length},
        input_components_{input_components},
        output_components_{output_components},
        is_inverse_{is_inverse},
        is_onesided_{is_onesided} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch", ProgramUniformVariableDataType::Uint32},
      {"signal_length", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"output_length", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t length_;
  uint32_t input_components_;
  uint32_t output_components_;
  bool is_inverse_;
  bool is_onesided_;
};

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
