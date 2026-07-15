// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

// Specialized depthwise 3x3 conv for NHWC layout.
// Each thread computes a 1x4 tile of output pixels along the width axis,
// sharing input reads across the tile. Kernel weights are loaded once per
// thread into registers. Only stride 1 and stride 2 are supported.
class DepthwiseConv3x3Program final : public Program<DepthwiseConv3x3Program> {
 public:
  DepthwiseConv3x3Program(const Activation& activation, bool has_bias, uint32_t stride)
      : Program{"DepthwiseConv3x3"}, activation_(activation), has_bias_(has_bias), stride_(stride) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"tiles_per_row", ProgramUniformVariableDataType::Uint32});

 private:
  const Activation& activation_;
  bool has_bias_;
  uint32_t stride_;
};

}  // namespace webgpu
}  // namespace onnxruntime
