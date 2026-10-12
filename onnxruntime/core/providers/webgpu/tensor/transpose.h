// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

// Transpose OIHW Weight to OHWI
#define WEBGPU_O_I_H_W2_O_H_W_I_PROGRAM_CONFIG(F)

struct OIHW2OHWIProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_O_I_H_W2_O_H_W_I_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "OIHW2OHWI";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_O_I_H_W2_O_H_W_I_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, O)                                             \
  U(Uint32, I)                                             \
  U(Uint32, H)                                             \
  U(Uint32, W)                                             \
  U(Uint32, Ci_tiles)                                      \
  U(Uint32, H_W_tiles)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_O_I_H_W2_O_H_W_I_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_O_I_H_W2_O_H_W_I_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_O_I_H_W2_O_H_W_I_PROGRAM_CONFIG

using OIHW2OHWIProgram = ConfiguredProgram<OIHW2OHWIProgramShader>;

class Transpose final : public WebGpuKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : WebGpuKernel{info}, TransposeBase{info} {
  }
  Status ComputeInternal(ComputeContext& context) const override;
  static Status DoTranspose(onnxruntime::webgpu::ComputeContextBase& context, gsl::span<const size_t> permutations, const Tensor& input, Tensor& output);

  // Tile edge, and how many of its rows one workgroup row covers. A 32-wide tile makes each
  // coalesced global access 64 bytes instead of 32, but a full 32x32 workgroup (1024 threads)
  // measured slower than 16x16; 32x8 with four rows per thread keeps the wider access and the
  // smaller workgroup.
  constexpr static uint32_t TILE_SIZE = 32;
  constexpr static uint32_t TILE_ROWS = 8;
};

#define WEBGPU_TRANSPOSE_PROGRAM_CONFIG(F) \
  F(InlinedVector<int64_t>, perm_)         \
  F(bool, use_shared_)

struct TransposeProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TRANSPOSE_PROGRAM_CONFIG);
  static Config MakeConfig(const gsl::span<const size_t>& permutations, bool use_shared) {
    Config result{};
    result.perm_ = InlinedVector<int64_t>(permutations.begin(), permutations.end());
    result.use_shared_ = bool(use_shared);

    return result;
  }
  static constexpr std::string_view name = "Transpose";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TRANSPOSE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TRANSPOSE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TRANSPOSE_PROGRAM_SHADER_UNIFORMS
  WEBGPU_PROGRAM_DEFINE_CONSTANTS({"tile_size", Transpose::TILE_SIZE},
                                  {"tile_rows", Transpose::TILE_ROWS});
};
#undef WEBGPU_TRANSPOSE_PROGRAM_CONFIG

using TransposeProgram = ConfiguredProgram<TransposeProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
