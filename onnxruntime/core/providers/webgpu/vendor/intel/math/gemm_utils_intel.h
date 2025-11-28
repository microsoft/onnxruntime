// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

void MatMulReadFnSourceIntel(ShaderHelper& shader,
                             const ShaderVariableHelper& a,
                             const ShaderVariableHelper& b,
                             const ShaderIndicesHelper* batch_dims,
                             bool transA,
                             bool transB,
                             bool is_vec4,
                             bool use_subgroup = false);

void MatMulWriteFnSourceIntel(ShaderHelper& shader,
                              const ShaderVariableHelper& output,
                              bool has_bias,
                              bool is_gemm,
                              int c_components,
                              int output_components,
                              bool c_is_scalar,
                              std::string activation_snippet = "",
                              bool is_channels_last = false);
}  // namespace webgpu
}  // namespace onnxruntime
