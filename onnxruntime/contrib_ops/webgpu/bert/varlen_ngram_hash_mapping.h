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

// Scans the whole cu_seqlens array once (single invocation) and writes a 1-element validity flag.
// Per-workgroup local checks in VarlenNGramHashMappingProgram (start < end, end <= total_tokens) are
// necessary but not sufficient: a single non-monotonic entry causes only the one workgroup that reads
// it to bail out, while neighboring workgroups whose own local check happens to still pass can claim
// overlapping output ranges (e.g. cu_seqlens = [0, 3, 2, 5]: request 1's start(3) >= end(2) check
// fails and is skipped, but request 0 writes [0, 3) and request 2 writes [2, 5), racing on token 2).
// VarlenNGramHashMappingProgram and VarlenNGramFillDefaultProgram both read this flag and their writes
// are mutually exclusive on it, so the output is always fully and unambiguously written regardless of
// launch order, as long as this program runs first (guaranteed by same-queue submission order).
class VarlenNGramValidateCuSeqlensProgram final : public Program<VarlenNGramValidateCuSeqlensProgram> {
 public:
  VarlenNGramValidateCuSeqlensProgram() : Program{"VarlenNGramValidateCuSeqlens"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_tokens", ProgramUniformVariableDataType::Uint32});
};

// Fills the hash_ids output (and, when present, present_ids) with deterministic defaults (zero hash
// ids, pad_id present_ids) when the validity flag produced by VarlenNGramValidateCuSeqlensProgram is
// false. Dispatched over a size derived only from host-known shape (max(total_tokens * num_heads,
// batch_size * state_length)), never from the untrusted cu_seqlens contents, so it always covers the
// whole output regardless of what cu_seqlens contains.
class VarlenNGramFillDefaultProgram final : public Program<VarlenNGramFillDefaultProgram> {
 public:
  VarlenNGramFillDefaultProgram(bool has_present_ids, bool has_eos_token_id)
      : Program{"VarlenNGramFillDefault"},
        has_present_ids_(has_present_ids),
        has_eos_token_id_(has_eos_token_id) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_count", ProgramUniformVariableDataType::Uint32},
                                          {"present_count", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  bool has_present_ids_;
  bool has_eos_token_id_;
};

// Computes n-gram hash ids over a packed, token-major batch of variable-length sequences. One
// workgroup is assigned per packed request (dispatch group count == batch_size) so the n-gram
// window for every token in that request is clamped at the request's own boundary and never reads
// across into an adjacent packed request.
class VarlenNGramHashMappingProgram final : public Program<VarlenNGramHashMappingProgram> {
 public:
  VarlenNGramHashMappingProgram(bool has_past_ids, bool has_head_offsets, bool has_eos_token_id,
                                bool has_segment_ids, bool reset_on_eos)
      : Program{"VarlenNGramHashMapping"},
        has_past_ids_(has_past_ids),
        has_head_offsets_(has_head_offsets),
        has_eos_token_id_(has_eos_token_id),
        has_segment_ids_(has_segment_ids),
        reset_on_eos_(reset_on_eos) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_tokens", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});

 private:
  bool has_past_ids_;
  bool has_head_offsets_;
  bool has_eos_token_id_;
  bool has_segment_ids_;
  bool reset_on_eos_;
};

// Emits the right-aligned trailing window of (past_ids ++ this request's tokens) per packed
// request, analogous to NGramPresentIdsProgram but scoped by cumulative_sequence_length instead of
// a fixed-stride batch row.
class VarlenNGramPresentIdsProgram final : public Program<VarlenNGramPresentIdsProgram> {
 public:
  VarlenNGramPresentIdsProgram(bool has_input_ids, bool has_past_ids, bool has_eos_token_id)
      : Program{"VarlenNGramPresentIds"},
        has_input_ids_(has_input_ids),
        has_past_ids_(has_past_ids),
        has_eos_token_id_(has_eos_token_id) {}
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
  bool has_eos_token_id_;
};

class VarlenNGramHashMapping final : public WebGpuKernel {
 public:
  explicit VarlenNGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t pad_id_;
  bool reset_on_eos_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
