// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class NGramHashMappingProgram final : public Program<NGramHashMappingProgram> {
 public:
  NGramHashMappingProgram(bool has_past_tokens, bool has_head_offsets, bool qwen_mode)
      : Program{"NGramHashMapping"},
        has_past_tokens_(has_past_tokens),
        has_head_offsets_(has_head_offsets),
        qwen_mode_(qwen_mode) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"context_length", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32},
                                          {"eos_token_id", ProgramUniformVariableDataType::Int32},
                                          {"reset_on_eos", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_tokens_;
  bool has_head_offsets_;
  bool qwen_mode_;
};

class NGramPresentTokensProgram final : public Program<NGramPresentTokensProgram> {
 public:
  explicit NGramPresentTokensProgram(bool has_past_tokens)
      : Program{"NGramPresentTokens"}, has_past_tokens_(has_past_tokens) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"context_length", ProgramUniformVariableDataType::Uint32},
                                          {"eos_token_id", ProgramUniformVariableDataType::Int32});

 private:
  bool has_past_tokens_;
};

class NGramHashMapping final : public WebGpuKernel {
 public:
  explicit NGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t pad_id_;
  int64_t eos_token_id_;
  bool reset_on_eos_;
  bool has_eos_token_id_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
