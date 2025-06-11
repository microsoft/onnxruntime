// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

void MatMulReadFnSource(ShaderHelper& shader,
                        const ShaderVariableHelper& a,
                        const ShaderVariableHelper& b,
                        const ShaderIndicesHelper* batch_dims,
                        bool transA,
                        bool transB,
                        bool is_vec4);

void MatMulWriteFnSource(ShaderHelper& shader,
                         const ShaderVariableHelper& output,
                         bool has_bias,
                         bool is_gemm,
                         int c_components,
                         int output_components,
                         bool c_is_scalar,
                         std::string activation_snippet = "",
                         bool is_channels_last = false);

// The two following functions are used to generate shader code for vec4 and scalar.
// It is used in GEMM, Matmul, and Conv.
Status MakeMatMulPackedVec4Source(ShaderHelper& shader,
                                  const InlinedVector<int64_t>& elements_per_thread,
                                  uint32_t workgroup_size_x,
                                  uint32_t workgroup_size_y,
                                  const std::string& data_type,
                                  const ShaderIndicesHelper* batch_dims,
                                  bool transpose_a = false,
                                  bool transpose_b = false,
                                  float alpha = 1.0f,
                                  bool need_handle_matmul = true,
                                  // When B is transposed, the components of output is might 1 though A and B is vec4.
                                  // e.g. A{32, 32}, B{33, 32} => Y{32, 33}
                                  int output_components = 4,
                                  uint32_t tile_inner = 32,
                                  bool split_k = false,
                                  uint32_t splitted_dim_inner = 32);

Status MakeMatMulPackedSource(ShaderHelper& shader,
                              const InlinedVector<int64_t>& elements_per_thread,
                              uint32_t workgroup_size_x,
                              uint32_t workgroup_size_y,
                              const std::string& data_type,
                              const ShaderIndicesHelper* batch_dims,
                              bool transpose_a = false,
                              bool transpose_b = false,
                              float alpha = 1.0f,
                              bool need_handle_matmul = true,
                              uint32_t tile_inner = 32,
                              bool split_k = false,
                              uint32_t splitted_dim_inner = 32);

}  // namespace webgpu
}  // namespace onnxruntime
