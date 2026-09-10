// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/sparse/sparse_attention_indexer_common.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class SparseAttentionIndexerFillProgram final : public Program<SparseAttentionIndexerFillProgram> {
 public:
  SparseAttentionIndexerFillProgram() : Program{"SparseAttentionIndexerFill"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32});
};

class SparseAttentionIndexerQsaConcatProgram final
    : public Program<SparseAttentionIndexerQsaConcatProgram> {
 public:
  SparseAttentionIndexerQsaConcatProgram(bool has_past, bool has_current)
      : Program{"SparseAttentionIndexerQsaConcat"}, has_past_{has_past}, has_current_{has_current} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_;
  bool has_current_;
};

class SparseAttentionIndexerQsaSelectProgram final
    : public Program<SparseAttentionIndexerQsaSelectProgram> {
 public:
  SparseAttentionIndexerQsaSelectProgram() : Program{"SparseAttentionIndexerQsaSelect"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});
};

class SparseAttentionIndexerCsaCopyCompressedProgram final
    : public Program<SparseAttentionIndexerCsaCopyCompressedProgram> {
 public:
  SparseAttentionIndexerCsaCopyCompressedProgram() : Program{"SparseAttentionIndexerCsaCopyCompressed"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"past_length", ProgramUniformVariableDataType::Uint32},
      {"present_length", ProgramUniformVariableDataType::Uint32});
};

class SparseAttentionIndexerCsaCompressProgram final
    : public Program<SparseAttentionIndexerCsaCompressProgram> {
 public:
  SparseAttentionIndexerCsaCompressProgram(bool has_past_buffer)
      : Program{"SparseAttentionIndexerCsaCompress"}, has_past_buffer_{has_past_buffer} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"work_items", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"past_compressed_length", ProgramUniformVariableDataType::Uint32},
      {"present_compressed_length", ProgramUniformVariableDataType::Uint32},
      {"past_buffer_length", ProgramUniformVariableDataType::Uint32},
      {"overlap_length", ProgramUniformVariableDataType::Uint32},
      {"new_window_count", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_past_buffer_;
};

class SparseAttentionIndexerCsaCopyBufferProgram final
    : public Program<SparseAttentionIndexerCsaCopyBufferProgram> {
 public:
  SparseAttentionIndexerCsaCopyBufferProgram(bool has_past_buffer, bool has_current)
      : Program{"SparseAttentionIndexerCsaCopyBuffer"},
        has_past_buffer_{has_past_buffer},
        has_current_{has_current} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"past_buffer_length", ProgramUniformVariableDataType::Uint32},
      {"present_buffer_length", ProgramUniformVariableDataType::Uint32},
      {"present_buffer_start", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_buffer_;
  bool has_current_;
};

class SparseAttentionIndexerCsaSelectProgram final
    : public Program<SparseAttentionIndexerCsaSelectProgram> {
 public:
  SparseAttentionIndexerCsaSelectProgram() : Program{"SparseAttentionIndexerCsaSelect"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"present_compressed_length", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"head_weight_scale", ProgramUniformVariableDataType::Float32});
};

class SparseAttentionIndexer final : public WebGpuKernel {
 public:
  explicit SparseAttentionIndexer(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  Status ComputeQsa(ComputeContext& context) const;
  Status ComputeCsa(ComputeContext& context) const;

  sparse_attention_indexer::Policy policy_;
  int64_t compress_ratio_;
  int64_t token_budget_;
  int64_t index_topk_;
  float epsilon_;
  float scale_;
  float head_weight_scale_;
  bool has_scale_;
  bool has_head_weight_scale_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
