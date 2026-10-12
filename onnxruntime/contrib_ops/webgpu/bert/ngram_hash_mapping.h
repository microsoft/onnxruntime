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

#define WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG(F) \
  F(bool, has_past_ids_)                             \
  F(bool, has_head_offsets_)                         \
  F(bool, has_eos_token_id_)                         \
  F(bool, has_segment_ids_)                          \
  F(bool, reset_on_eos_)

struct NGramHashMappingProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past_ids, bool has_head_offsets, bool has_eos_token_id, bool has_segment_ids, bool reset_on_eos) {
    Config result{};
    result.has_past_ids_ = bool(has_past_ids);
    result.has_head_offsets_ = bool(has_head_offsets);
    result.has_eos_token_id_ = bool(has_eos_token_id);
    result.has_segment_ids_ = bool(has_segment_ids);
    result.reset_on_eos_ = bool(reset_on_eos);

    return result;
  }
  static constexpr std::string_view name = "NGramHashMapping";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                            \
  U(Uint32, sequence_length)                                  \
  U(Uint32, max_ngram_size)                                   \
  U(Uint32, n_head_per_ngram)                                 \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG

using NGramHashMappingProgram = ConfiguredProgram<NGramHashMappingProgramShader>;

#define WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG(F) \
  F(bool, has_input_ids_)                           \
  F(bool, has_past_ids_)                            \
  F(bool, has_eos_token_id_)                        \
  F(bool, past_aliases_present_)

struct NGramPresentIdsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_input_ids, bool has_past_ids, bool has_eos_token_id, bool past_aliases_present) {
    Config result{};
    result.has_input_ids_ = bool(has_input_ids);
    result.has_past_ids_ = bool(has_past_ids);
    result.has_eos_token_id_ = bool(has_eos_token_id);
    result.past_aliases_present_ = bool(past_aliases_present);

    return result;
  }
  static constexpr std::string_view name = "NGramPresentIds";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                      \
  U(Uint32, sequence_length)                                 \
  U(Uint32, state_length)                                    \
  U(Int32, pad_id)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG

using NGramPresentIdsProgram = ConfiguredProgram<NGramPresentIdsProgramShader>;

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
