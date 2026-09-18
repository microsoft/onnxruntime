// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class BranchwiseRMSNormProgram final
    : public Program<BranchwiseRMSNormProgram> {
 public:
  BranchwiseRMSNormProgram(bool has_scale, bool shared_scale)
      : Program{"BranchwiseRMSNorm"},
        has_scale_(has_scale),
        shared_scale_(shared_scale) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"groups", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_scale_;
  bool shared_scale_;
};

class BranchwiseRMSNorm final : public WebGpuKernel {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
  int64_t num_branches_;
};

class ScaledSiLUProgram final : public Program<ScaledSiLUProgram> {
 public:
  explicit ScaledSiLUProgram(bool has_scale)
      : Program{"ScaledSiLU"}, has_scale_(has_scale) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});

 private:
  bool has_scale_;
};

class ScaledSiLU final : public WebGpuKernel {
 public:
  explicit ScaledSiLU(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float alpha_;
};

class HyperConnectionPreMixProgram final
    : public Program<HyperConnectionPreMixProgram> {
 public:
  explicit HyperConnectionPreMixProgram(int gate_layout)
      : Program{"HyperConnectionPreMix"}, gate_layout_(gate_layout) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"reduction_scale", ProgramUniformVariableDataType::Float32});

 private:
  int gate_layout_;
};

class HyperConnectionPreMix final : public WebGpuKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

class HyperConnectionPostMixProgram final
    : public Program<HyperConnectionPostMixProgram> {
 public:
  HyperConnectionPostMixProgram(int gate_layout, bool has_stream_mix)
      : Program{"HyperConnectionPostMix"},
        gate_layout_(gate_layout),
        has_stream_mix_(has_stream_mix) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32});

 private:
  int gate_layout_;
  bool has_stream_mix_;
};

class HyperConnectionPostMix final : public WebGpuKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib::webgpu
