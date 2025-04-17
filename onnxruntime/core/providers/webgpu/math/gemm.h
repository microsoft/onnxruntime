// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class GemmProgram final : public Program<GemmProgram> {
 public:
  GemmProgram(bool transA, bool transB, float alpha, bool need_handle_bias, bool need_handle_matmul)
      : Program{"Gemm"},
        transA_{transA},
        transB_{transB},
        alpha_{alpha},
        need_handle_bias_{need_handle_bias},
        need_handle_matmul_{need_handle_matmul} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_tile_n", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  bool need_handle_bias_;
  bool need_handle_matmul_;
};

class Gemm final : public WebGpuKernel {
 public:
  Gemm(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t transA_temp;
    info.GetAttrOrDefault("transA", &transA_temp, static_cast<int64_t>(0));
    transA_ = transA_temp != 0;

    int64_t transB_temp;
    info.GetAttrOrDefault("transB", &transB_temp, static_cast<int64_t>(0));
    transB_ = transB_temp != 0;

    info.GetAttrOrDefault("alpha", &alpha_, 1.0f);
    info.GetAttrOrDefault("beta", &beta_, 1.0f);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  float beta_;
};

}  // namespace webgpu
}  // namespace onnxruntime
