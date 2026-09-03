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

// Computes n-gram hash ids over a packed, token-major batch of variable-length sequences. One
// workgroup is assigned per packed request (dispatch group count == batch_size) so the n-gram
// window for every token in that request is clamped at the request's own boundary and never reads
// across into an adjacent packed request.
class VarlenNGramHashMappingProgram final : public Program<VarlenNGramHashMappingProgram> {
 public:
  explicit VarlenNGramHashMappingProgram(bool has_past_ids)
      : Program{"VarlenNGramHashMapping"}, has_past_ids_(has_past_ids) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_tokens", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  bool has_past_ids_;
};

// Emits the right-aligned trailing window of (past_ids ++ this request's tokens) per packed
// request, analogous to NGramPresentIdsProgram but scoped by cumulative_sequence_length instead of
// a fixed-stride batch row.
class VarlenNGramPresentIdsProgram final : public Program<VarlenNGramPresentIdsProgram> {
 public:
  VarlenNGramPresentIdsProgram(bool has_input_ids, bool has_past_ids)
      : Program{"VarlenNGramPresentIds"}, has_input_ids_(has_input_ids), has_past_ids_(has_past_ids) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"state_length", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_tokens", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  // False when total_tokens == 0. WebGPU cannot bind a zero-sized buffer, and in that case every
  // present slot is history or pad_id, so the input_ids branch is omitted entirely.
  bool has_input_ids_;
  bool has_past_ids_;
};

class VarlenNGramHashMapping final : public WebGpuKernel {
 public:
  explicit VarlenNGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t pad_id_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
