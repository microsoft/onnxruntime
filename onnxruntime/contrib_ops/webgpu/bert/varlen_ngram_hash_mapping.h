// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
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
#define WEBGPU_VARLEN_N_GRAM_VALIDATE_CU_SEQLENS_PROGRAM_CONFIG(F)

struct VarlenNGramValidateCuSeqlensProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_VALIDATE_CU_SEQLENS_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramValidateCuSeqlens";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_VALIDATE_CU_SEQLENS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                                     \
  U(Uint32, total_tokens)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_VALIDATE_CU_SEQLENS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_VALIDATE_CU_SEQLENS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_VALIDATE_CU_SEQLENS_PROGRAM_CONFIG

using VarlenNGramValidateCuSeqlensProgram = ConfiguredProgram<VarlenNGramValidateCuSeqlensProgramShader>;

// Fills the hash_ids output (and, when present, present_ids) with deterministic defaults (zero hash
// ids, pad_id present_ids) when the validity flag produced by VarlenNGramValidateCuSeqlensProgram is
// false. Dispatched over a size derived only from host-known shape (max(total_tokens * num_heads,
// batch_size * state_length)), never from the untrusted cu_seqlens contents, so it always covers the
// whole output regardless of what cu_seqlens contains.
#define WEBGPU_VARLEN_N_GRAM_FILL_DEFAULT_PROGRAM_CONFIG(F) \
  F(bool, has_present_ids_)                                 \
  F(bool, has_present_segment_ids_)                         \
  F(bool, has_state_update_)                                \
  F(bool, has_eos_token_id_)

struct VarlenNGramFillDefaultProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_FILL_DEFAULT_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_present_ids, bool has_present_segment_ids, bool has_state_update, bool has_eos_token_id) {
    Config result{};
    result.has_present_ids_ = bool(has_present_ids);
    result.has_present_segment_ids_ = bool(has_present_segment_ids);
    result.has_state_update_ = bool(has_state_update);
    result.has_eos_token_id_ = bool(has_eos_token_id);

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramFillDefault";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_FILL_DEFAULT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_count)                                            \
  U(Uint32, present_count)                                           \
  U(Uint32, state_update_count)                                      \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_FILL_DEFAULT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_FILL_DEFAULT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_FILL_DEFAULT_PROGRAM_CONFIG

using VarlenNGramFillDefaultProgram = ConfiguredProgram<VarlenNGramFillDefaultProgramShader>;

// Computes n-gram hash ids over a packed, token-major batch of variable-length sequences. One
// workgroup is assigned per packed request (dispatch group count == batch_size) so the n-gram
// window for every token in that request is clamped at the request's own boundary and never reads
// across into an adjacent packed request.
#define WEBGPU_VARLEN_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG(F) \
  F(bool, has_past_ids_)                                    \
  F(bool, has_eos_token_id_)                                \
  F(bool, has_nearest_reset_)

struct VarlenNGramHashMappingProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past_ids, bool has_eos_token_id, bool has_nearest_reset) {
    Config result{};
    result.has_past_ids_ = bool(has_past_ids);
    result.has_eos_token_id_ = bool(has_eos_token_id);
    result.has_nearest_reset_ = bool(has_nearest_reset);

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramHashMapping";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_HASH_MAPPING_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                              \
  U(Uint32, total_tokens)                                            \
  U(Uint32, max_ngram_size)                                          \
  U(Uint32, n_head_per_ngram)                                        \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_HASH_MAPPING_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_HASH_MAPPING_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG

using VarlenNGramHashMappingProgram = ConfiguredProgram<VarlenNGramHashMappingProgramShader>;

#define WEBGPU_VARLEN_N_GRAM_PREPARE_NEAREST_RESET_PROGRAM_CONFIG(F) \
  F(bool, has_past_ids_)                                             \
  F(bool, has_eos_token_id_)                                         \
  F(bool, has_segment_ids_)                                          \
  F(bool, has_past_segment_ids_)                                     \
  F(bool, reset_on_eos_)

struct VarlenNGramPrepareNearestResetProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_PREPARE_NEAREST_RESET_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past_ids, bool has_eos_token_id, bool has_segment_ids, bool has_past_segment_ids, bool reset_on_eos) {
    Config result{};
    result.has_past_ids_ = bool(has_past_ids);
    result.has_eos_token_id_ = bool(has_eos_token_id);
    result.has_segment_ids_ = bool(has_segment_ids);
    result.has_past_segment_ids_ = bool(has_past_segment_ids);
    result.reset_on_eos_ = bool(reset_on_eos);

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramPrepareNearestReset";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_PREPARE_NEAREST_RESET_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                                       \
  U(Uint32, total_tokens)                                                     \
  U(Uint32, max_ngram_size)                                                   \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_PREPARE_NEAREST_RESET_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_PREPARE_NEAREST_RESET_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_PREPARE_NEAREST_RESET_PROGRAM_CONFIG

using VarlenNGramPrepareNearestResetProgram = ConfiguredProgram<VarlenNGramPrepareNearestResetProgramShader>;

#define WEBGPU_VARLEN_N_GRAM_ADD_HEAD_OFFSETS_PROGRAM_CONFIG(F)

struct VarlenNGramAddHeadOffsetsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_ADD_HEAD_OFFSETS_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramAddHeadOffsets";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_ADD_HEAD_OFFSETS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_count)                                                \
  U(Uint32, num_heads)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_ADD_HEAD_OFFSETS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_ADD_HEAD_OFFSETS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_ADD_HEAD_OFFSETS_PROGRAM_CONFIG

using VarlenNGramAddHeadOffsetsProgram = ConfiguredProgram<VarlenNGramAddHeadOffsetsProgramShader>;

// Emits the right-aligned trailing window of (past_ids ++ this request's tokens) per packed
// request, analogous to NGramPresentIdsProgram but scoped by cumulative_sequence_length instead of
// a fixed-stride batch row.
#define WEBGPU_VARLEN_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG(F) \
  F(bool, has_present_ids_)                                \
  F(bool, has_present_segment_ids_)                        \
  F(bool, has_past_ids_)                                   \
  F(bool, has_past_segment_ids_)                           \
  F(bool, has_eos_token_id_)

struct VarlenNGramPresentIdsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_present_ids, bool has_present_segment_ids, bool has_past_ids, bool has_past_segment_ids,
                           bool has_eos_token_id) {
    Config result{};
    result.has_present_ids_ = bool(has_present_ids);
    result.has_present_segment_ids_ = bool(has_present_segment_ids);
    result.has_past_ids_ = bool(has_past_ids);
    result.has_past_segment_ids_ = bool(has_past_segment_ids);
    result.has_eos_token_id_ = bool(has_eos_token_id);

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramPresentIds";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_PRESENT_IDS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                                  \
  U(Uint32, state_length)                                           \
  U(Uint32, batch_size)                                             \
  U(Uint32, total_tokens)                                           \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_PRESENT_IDS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_PRESENT_IDS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG

using VarlenNGramPresentIdsProgram = ConfiguredProgram<VarlenNGramPresentIdsProgramShader>;

#define WEBGPU_VARLEN_N_GRAM_STATE_UPDATE_PROGRAM_CONFIG(F) \
  F(bool, has_past_ids_)                                    \
  F(bool, has_eos_token_id_)

struct VarlenNGramStateUpdateProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_VARLEN_N_GRAM_STATE_UPDATE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past_ids, bool has_eos_token_id) {
    Config result{};
    result.has_past_ids_ = bool(has_past_ids);
    result.has_eos_token_id_ = bool(has_eos_token_id);

    return result;
  }
  static constexpr std::string_view name = "VarlenNGramStateUpdate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_VARLEN_N_GRAM_STATE_UPDATE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                                   \
  U(Uint32, state_length)                                            \
  U(Uint32, state_update_capacity)                                   \
  U(Uint32, total_tokens)                                            \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_VARLEN_N_GRAM_STATE_UPDATE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_VARLEN_N_GRAM_STATE_UPDATE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_VARLEN_N_GRAM_STATE_UPDATE_PROGRAM_CONFIG

using VarlenNGramStateUpdateProgram = ConfiguredProgram<VarlenNGramStateUpdateProgramShader>;

class VarlenNGramHashMapping final : public WebGpuKernel {
 public:
  explicit VarlenNGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  int64_t state_update_capacity_;
  int64_t pad_id_;
  bool reset_on_eos_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
