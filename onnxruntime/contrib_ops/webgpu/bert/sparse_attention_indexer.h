// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/sparse/sparse_attention_indexer_common.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ConfiguredProgram;
using onnxruntime::webgpu::ConfiguredShaderHelper;
using onnxruntime::webgpu::ProgramUniformVariableDataType;
using onnxruntime::webgpu::WebGpuKernel;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_CONFIG(F)

struct SparseAttentionIndexerFillProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerFill";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_CONFIG

using SparseAttentionIndexerFillProgram = ConfiguredProgram<SparseAttentionIndexerFillProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_CONFIG(F) \
  F(bool, has_past_)                                                 \
  F(bool, has_current_)

struct SparseAttentionIndexerQsaConcatProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_CONFIG);
    Config(bool has_past, bool has_current) : has_past_{has_past}, has_current_{has_current} {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerQsaConcat";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"key_row_stride", ProgramUniformVariableDataType::Uint32},
      {"key_offset", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_CONFIG

using SparseAttentionIndexerQsaConcatProgram = ConfiguredProgram<SparseAttentionIndexerQsaConcatProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG(F) F(bool, has_mask_)

struct SparseAttentionIndexerQsaSelectProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG);
    Config(bool has_mask) : has_mask_{has_mask} {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerQsaSelect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_row_stride", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"rotary_cache_batch_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG

using SparseAttentionIndexerQsaSelectProgram = ConfiguredProgram<SparseAttentionIndexerQsaSelectProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_CONFIG(F)

struct SparseAttentionIndexerCsaCopyCompressedProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerCsaCopyCompressed";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"past_length", ProgramUniformVariableDataType::Uint32},
      {"present_length", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_CONFIG

using SparseAttentionIndexerCsaCopyCompressedProgram =
    ConfiguredProgram<SparseAttentionIndexerCsaCopyCompressedProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_CONFIG(F) F(bool, has_past_buffer_)

struct SparseAttentionIndexerCsaCompressProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_CONFIG);
    Config(bool has_past_buffer) : has_past_buffer_{has_past_buffer} {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerCsaCompress";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"work_items", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"rotary_cache_batch_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"past_compressed_length", ProgramUniformVariableDataType::Uint32},
      {"present_compressed_length", ProgramUniformVariableDataType::Uint32},
      {"past_buffer_length", ProgramUniformVariableDataType::Uint32},
      {"overlap_length", ProgramUniformVariableDataType::Uint32},
      {"new_window_count", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_CONFIG

using SparseAttentionIndexerCsaCompressProgram = ConfiguredProgram<SparseAttentionIndexerCsaCompressProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_CONFIG(F) \
  F(bool, has_past_buffer_)                                               \
  F(bool, has_current_)

struct SparseAttentionIndexerCsaCopyBufferProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_CONFIG);
    Config(bool has_past_buffer, bool has_current) : has_past_buffer_{has_past_buffer}, has_current_{has_current} {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerCsaCopyBuffer";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"past_buffer_length", ProgramUniformVariableDataType::Uint32},
      {"present_buffer_length", ProgramUniformVariableDataType::Uint32},
      {"present_buffer_start", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_CONFIG

using SparseAttentionIndexerCsaCopyBufferProgram = ConfiguredProgram<SparseAttentionIndexerCsaCopyBufferProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG(F)

struct SparseAttentionIndexerCsaSelectProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "SparseAttentionIndexerCsaSelect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"rotary_cache_batch_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"present_compressed_length", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32},
      {"head_weight_scale", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG

using SparseAttentionIndexerCsaSelectProgram = ConfiguredProgram<SparseAttentionIndexerCsaSelectProgramShader>;

class SparseAttentionIndexer final : public WebGpuKernel {
 public:
  explicit SparseAttentionIndexer(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  Status ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const;
  Status ComputeCsa(onnxruntime::webgpu::ComputeContext& context) const;

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
