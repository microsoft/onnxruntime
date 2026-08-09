// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webgpu {

class ConvTranspose2DProgram : public Program<ConvTranspose2DProgram> {
 public:
  ConvTranspose2DProgram(bool is_channels_last, bool has_bias, uint32_t components, uint32_t a_components, uint32_t b_components, uint32_t input_channels_remainder, bool pack_input_as4) : Program("ConvTranspose2D"), is_channels_last_(is_channels_last), has_bias_(has_bias), components_(components), a_components_(a_components), b_components_(b_components), input_channels_remainder_(input_channels_remainder), pack_input_as4_(pack_input_as4) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"filter_dims", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"effective_filter_dims", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"input_channels_per_group_int", ProgramUniformVariableDataType::Uint32},
      {"input_channels_per_group", ProgramUniformVariableDataType::Uint32},
      {"output_channels_per_group", ProgramUniformVariableDataType::Uint32});

 private:
  bool is_channels_last_;
  bool has_bias_;
  uint32_t components_;
  uint32_t a_components_;
  uint32_t b_components_;
  uint32_t input_channels_remainder_;
  bool pack_input_as4_;
};

ConvTranspose2DProgram CreateConvTranspose2DProgram(const std::vector<const Tensor*>& inputs, const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides, const std::vector<uint32_t>& dilations, Tensor* output, bool is_channels_last, const std::vector<TensorShape>& modified_input_output_shapes, uint32_t groups);

}  // namespace webgpu
}  // namespace onnxruntime
