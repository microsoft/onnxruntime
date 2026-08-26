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
  NGramHashMappingProgram(bool has_past_tokens, bool has_head_offsets, bool has_eos_token_id,
                          bool has_segment_ids, bool has_present_tokens, bool reset_on_eos)
      : Program{"NGramHashMapping"},
        has_past_tokens_(has_past_tokens),
        has_head_offsets_(has_head_offsets),
        has_eos_token_id_(has_eos_token_id),
        has_segment_ids_(has_segment_ids),
        has_present_tokens_(has_present_tokens),
        reset_on_eos_(reset_on_eos) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"main_total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"history_length", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  bool has_past_tokens_;
  bool has_head_offsets_;
  bool has_eos_token_id_;
  bool has_segment_ids_;
  bool has_present_tokens_;
  bool reset_on_eos_;
};

class NGramHashMapping final : public WebGpuKernel {
 public:
  explicit NGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t pad_id_;
  int64_t reset_on_eos_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
