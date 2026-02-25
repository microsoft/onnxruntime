// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

class Im2ColMatMulProgram final : public Program<Im2ColMatMulProgram> {
 public:
  Im2ColMatMulProgram(bool has_bias,
                      uint32_t tile_m,
                      uint32_t tile_n,
                      bool use_subgroup) : Program("Im2ColMatMul"),
                                           has_bias_(has_bias),
                                           tile_m_(tile_m),
                                           tile_n_(tile_n),
                                           use_subgroup_(use_subgroup) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

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
      {"strides", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;

  uint32_t tile_m_;
  uint32_t tile_n_;
  bool use_subgroup_;
};

bool CanApplyIm2ColMatMulProgram(ComputeContextBase& context,
                                 const bool is_channels_last,
                                 const bool is_fused,
                                 const TensorShape kernel_shape,
                                 const uint32_t group);

Status ApplyIm2ColMatMulProgram(ComputeContext& context,
                                const bool is_channels_last,
                                const std::vector<uint32_t>& dilations,
                                const std::vector<uint32_t>& pads,
                                const std::vector<uint32_t>& strides,
                                Tensor* output);

}  // namespace webgpu
}  // namespace onnxruntime
