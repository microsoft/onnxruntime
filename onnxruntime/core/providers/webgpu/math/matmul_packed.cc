// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_packed.h"

#include "core/providers/webgpu/math/gemm_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include <string>
namespace onnxruntime {
namespace webgpu {

Status MatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  // declare the read and write functions
  MatMulReadFnSource(shader, a, b, &batch_dims, /*transA = */ false, /*transB = */ false, is_vec4_);
  MatMulWriteFnSource(shader, output, has_bias_, /* is_gemm = */ false, 1, is_vec4_ ? 4 : 1, false, apply_activation, is_channels_last_);
  std::string data_type = "a_element_t";
  // generate the main function
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims));
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
