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
  explicit NGramHashMappingProgram(bool has_past_ids)
      : Program{"NGramHashMapping"}, has_past_ids_(has_past_ids) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  bool has_past_ids_;
};

// Emits the right-aligned trailing window of (past_ids ++ input_ids) so the next call can continue
// the n-gram windows across invocations.
class NGramPresentIdsProgram final : public Program<NGramPresentIdsProgram> {
 public:
  NGramPresentIdsProgram(bool has_input_ids, bool has_past_ids)
      : Program{"NGramPresentIds"}, has_input_ids_(has_input_ids), has_past_ids_(has_past_ids) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"state_length", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  // False when sequence_length == 0. WebGPU cannot bind a zero-sized buffer, and in that case every
  // present slot is history or pad_id, so the input_ids branch is omitted entirely.
  bool has_input_ids_;
  bool has_past_ids_;
};

class NGramHashMapping final : public WebGpuKernel {
 public:
  explicit NGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t pad_id_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
