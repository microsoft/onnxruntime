// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

class GemmVec4Program final : public Program<GemmVec4Program> {
 public:
  GemmVec4Program(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul, int c_components, bool c_is_scalar, int output_components)
      : Program{"GemmVec4"},
        transA_{transA},
        transB_{transB},
        alpha_{alpha},
        need_handle_bias_{need_handle_bias},
        need_handle_matmul_{need_handle_matmul},
        c_components_(c_components),
        c_is_scalar_(c_is_scalar),
        output_components_(output_components) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  void MatMulReadFnSource(ShaderHelper& shader) const;
  void MatMulWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& output) const;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_tile_n", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"M4", ProgramUniformVariableDataType::Uint32},
      {"N4", ProgramUniformVariableDataType::Uint32},
      {"K4", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  bool need_handle_bias_;
  bool need_handle_matmul_;
  int c_components_;
  bool c_is_scalar_ = false;
  int output_components_;
};

Status ApplyGemmVec4(const Tensor* a,
                     const Tensor* b,
                     const Tensor* c,
                     bool transA,
                     bool transB,
                     float alpha,
                     float beta,
                     ComputeContext& context,
                     Tensor* y);

bool CanApplyGemmVec4(const Tensor* a,
                      const Tensor* b);

}  // namespace webgpu
}  // namespace onnxruntime
