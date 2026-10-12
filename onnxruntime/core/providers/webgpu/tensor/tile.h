// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_TILE_PROGRAM_CONFIG(F) F(bool, is_int64_)

struct TileProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TILE_PROGRAM_CONFIG);
  static Config MakeConfig(bool is_int64) {
    Config result{};
    result.is_int64_ = bool{is_int64};

    return result;
  }
  static constexpr std::string_view name = "Tile";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TILE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                       \
  U(Uint32, repeats)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TILE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TILE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TILE_PROGRAM_CONFIG

using TileProgram = ConfiguredProgram<TileProgramShader>;

class Tile final : public WebGpuKernel {
 public:
  Tile(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

// Create Tile kernel info with appropriate type constraints based on int64 support.
// Tile is a pure data-movement op; int64 is safe because element values are never
// interpreted or used in shader arithmetic.
KernelCreateInfo CreateTileVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateTileKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
