// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_IM2_COL_MAT_MUL_PROGRAM_CONFIG(F) \
  F(bool, has_bias_)                             \
  F(uint32_t, tile_m_)                           \
  F(uint32_t, tile_n_)                           \
  F(uint32_t, vec_size_)                         \
  F(bool, use_subgroup_)                         \
  F(ShaderActivation, activation_)

struct Im2ColMatMulProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_IM2_COL_MAT_MUL_PROGRAM_CONFIG);
    Config(bool has_bias, uint32_t tile_m, uint32_t tile_n, uint32_t vec_size, bool use_subgroup,
           const Activation& activation)
        : has_bias_(has_bias),
          tile_m_(tile_m),
          tile_n_(tile_n),
          vec_size_(vec_size),
          use_subgroup_(use_subgroup),
          activation_(activation) {}
  };
  static constexpr std::string_view name = "Im2ColMatMul";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch", ProgramUniformVariableDataType::Uint32},
      {"src_h", ProgramUniformVariableDataType::Uint32},
      {"src_w", ProgramUniformVariableDataType::Uint32},
      {"channel_i", ProgramUniformVariableDataType::Uint32},
      {"kernel_h", ProgramUniformVariableDataType::Uint32},
      {"kernel_w", ProgramUniformVariableDataType::Uint32},
      {"output_h", ProgramUniformVariableDataType::Uint32},
      {"output_w", ProgramUniformVariableDataType::Uint32},
      {"im2col_m", ProgramUniformVariableDataType::Uint32},
      {"im2col_k", ProgramUniformVariableDataType::Uint32},
      {"im2col_n", ProgramUniformVariableDataType::Uint32},
      {"M_tiles", ProgramUniformVariableDataType::Uint32},
      {"N_tiles", ProgramUniformVariableDataType::Uint32},
      {"K_tiles", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);
};
#undef WEBGPU_IM2_COL_MAT_MUL_PROGRAM_CONFIG

using Im2ColMatMulProgram = ConfiguredProgram<Im2ColMatMulProgramShader>;

bool CanApplyIm2ColMatMulProgram(ComputeContextBase& context,
                                 const bool is_channels_last,
                                 const Activation& activation,
                                 const TensorShape kernel_shape,
                                 const uint32_t group,
                                 const MLDataType data_type);

// Transposes the OIHW weight into the OHWI layout expected by Im2ColMatMulProgram.
// Called from Conv::PrePackInternal so the transpose runs once at session
// initialization instead of on every inference.
Status PrePackIm2ColMatMulWeight(ComputeContextBase& context,
                                 const Tensor& weight,
                                 AllocatorPtr alloc,
                                 /*out*/ std::unique_ptr<Tensor>& packed_weight);

// `packed_weight` is the OHWI weight produced by PrePackIm2ColMatMulWeight. When it
// is nullptr, the OIHW weight is read from input 1 and transposed on the fly.
Status ApplyIm2ColMatMulProgram(ComputeContext& context,
                                const bool is_channels_last,
                                const Activation& activation,
                                const std::vector<uint32_t>& dilations,
                                const std::vector<uint32_t>& pads,
                                const std::vector<uint32_t>& strides,
                                const Tensor* packed_weight,
                                Tensor* output);

}  // namespace webgpu
}  // namespace onnxruntime
