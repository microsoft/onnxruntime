// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ComputeContext;

class ShortConvProgram final : public Program<ShortConvProgram> {
 public:
  ShortConvProgram(bool has_bias, bool apply_silu) : Program{"ShortConv"}, has_bias_(has_bias), apply_silu_(apply_silu) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_size", ProgramUniformVariableDataType::Uint32},
                                          {"dilation", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_bias_;
  bool apply_silu_;
};

class ShortConv final : public WebGpuKernel {
 public:
  explicit ShortConv(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::string activation_;
  int64_t dilation_;
  float epsilon_;
};

class NgramHashMappingProgram final : public Program<NgramHashMappingProgram> {
 public:
  NgramHashMappingProgram() : Program{"NgramHashMapping"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});
};

class NgramHashMapping final : public WebGpuKernel {
 public:
  explicit NgramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t pad_id_;
};

class EngramGateProgram final : public Program<EngramGateProgram> {
 public:
  EngramGateProgram(bool has_key_bias, bool has_value_bias)
      : Program{"EngramGate"}, has_key_bias_(has_key_bias), has_value_bias_(has_value_bias) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"hc_mult", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"embedding_size", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_key_bias_;
  bool has_value_bias_;
};

class EngramGate final : public WebGpuKernel {
 public:
  explicit EngramGate(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
