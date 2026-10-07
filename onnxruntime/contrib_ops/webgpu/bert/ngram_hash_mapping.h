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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG);
    Config(bool has_past_ids, bool has_head_offsets, bool has_eos_token_id, bool has_segment_ids, bool reset_on_eos)
        : has_past_ids_(has_past_ids),
          has_head_offsets_(has_head_offsets),
          has_eos_token_id_(has_eos_token_id),
          has_segment_ids_(has_segment_ids),
          reset_on_eos_(reset_on_eos) {}
  };
  static constexpr std::string_view name = "NGramHashMapping";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"max_ngram_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_head_per_ngram", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_N_GRAM_HASH_MAPPING_PROGRAM_CONFIG

using NGramHashMappingProgram = ConfiguredProgram<NGramHashMappingProgramShader>;

#define WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG(F) \
  F(bool, has_input_ids_)                           \
  F(bool, has_past_ids_)                            \
  F(bool, has_eos_token_id_)                        \
  F(bool, past_aliases_present_)

struct NGramPresentIdsProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_N_GRAM_PRESENT_IDS_PROGRAM_CONFIG);
    Config(bool has_input_ids, bool has_past_ids, bool has_eos_token_id, bool past_aliases_present)
        : has_input_ids_(has_input_ids),
          has_past_ids_(has_past_ids),
          has_eos_token_id_(has_eos_token_id),
          past_aliases_present_(past_aliases_present) {}
  };
  static constexpr std::string_view name = "NGramPresentIds";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"state_length", ProgramUniformVariableDataType::Uint32},
                                          {"pad_id", ProgramUniformVariableDataType::Int32});
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
