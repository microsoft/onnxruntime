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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TILE_PROGRAM_CONFIG);
    Config(bool is_int64) : is_int64_{is_int64} {}
  };
  static constexpr std::string_view name = "Tile";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"repeats", ProgramUniformVariableDataType::Uint32});
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
