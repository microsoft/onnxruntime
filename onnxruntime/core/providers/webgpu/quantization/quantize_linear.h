// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// How the quantized input is packed into u32 words.
enum class PackingMode {
  None,     // no packing (e.g. int32)
  Packed8,  // 8-bit: 4 elements per u32, uses unpack4x[I/U]8
  Packed4,  // 4-bit: 8 elements per u32, manual bit extraction
};

#define WEBGPU_DEQUANTIZE_LINEAR_PROGRAM_CONFIG(F) \
  F(PackingMode, packing_)                         \
  F(bool, packed_signed_)                          \
  F(bool, per_layer_)                              \
  F(bool, per_axis_)                               \
  F(bool, has_zeropoint_)                          \
  F(int, rank_)

struct DequantizeLinearProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_DEQUANTIZE_LINEAR_PROGRAM_CONFIG);
    Config(PackingMode packing, bool is_packed_signed, bool per_layer, bool per_axis, bool has_zeropoint, int rank = 0)
        : packing_{packing},
          packed_signed_{is_packed_signed},
          per_layer_{per_layer},
          per_axis_{per_axis},
          has_zeropoint_{has_zeropoint},
          rank_{rank} {}
  };
  static constexpr std::string_view name = "DequantizeLinear";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"axis", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_DEQUANTIZE_LINEAR_PROGRAM_CONFIG

using DequantizeLinearProgram = ConfiguredProgram<DequantizeLinearProgramShader>;

class DequantizeLinear final : public WebGpuKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : WebGpuKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
    block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 0);
    output_dtype_ = info.GetAttrOrDefault<int64_t>("output_dtype", 0);
    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
  int64_t output_dtype_;
};

}  // namespace webgpu
}  // namespace onnxruntime
