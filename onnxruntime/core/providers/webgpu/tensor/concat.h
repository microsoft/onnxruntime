// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CONCAT_PROGRAM_CONFIG(F) \
  F(size_t, axis_)                      \
  F(bool, is_int64_)

struct ConcatProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_CONCAT_PROGRAM_CONFIG);
  static Config MakeConfig(size_t axis, bool is_int64) {
    Config result{};
    result.axis_ = size_t{axis};
    result.is_int64_ = bool{is_int64};

    return result;
  }
  static constexpr std::string_view name = "Concat";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_CONCAT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, offsets)                             \
  U(Uint32, sizes_in_concat_axis)                \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_CONCAT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_CONCAT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_CONCAT_PROGRAM_CONFIG

using ConcatProgram = ConfiguredProgram<ConcatProgramShader>;

class Concat final : public WebGpuKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : WebGpuKernel(info), ConcatBase(info) {
  }

  Status ComputeInternal(ComputeContext& context) const override;
};

// Create Concat kernel info with appropriate type constraints based on int64 support
KernelCreateInfo CreateConcatVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateConcatKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
