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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerFill";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_FILL_PROGRAM_CONFIG

using SparseAttentionIndexerFillProgram = ConfiguredProgram<SparseAttentionIndexerFillProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_CONFIG(F) \
  F(bool, has_past_)                                                 \
  F(bool, has_current_)

struct SparseAttentionIndexerQsaConcatProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past, bool has_current) {
    Config result{};
    result.has_past_ = bool{has_past};
    result.has_current_ = bool{has_current};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerQsaConcat";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                                            \
  U(Uint32, sequence_length)                                                  \
  U(Uint32, past_sequence_length)                                             \
  U(Uint32, total_sequence_length)                                            \
  U(Uint32, head_size)                                                        \
  U(Uint32, key_row_stride)                                                   \
  U(Uint32, key_offset)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_CONCAT_PROGRAM_CONFIG

using SparseAttentionIndexerQsaConcatProgram = ConfiguredProgram<SparseAttentionIndexerQsaConcatProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG(F) F(bool, has_mask_)

struct SparseAttentionIndexerQsaSelectProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_mask) {
    Config result{};
    result.has_mask_ = bool{has_mask};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerQsaSelect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, rows)                                                             \
  U(Uint32, sequence_length)                                                  \
  U(Uint32, num_heads)                                                        \
  U(Uint32, head_size)                                                        \
  U(Uint32, query_row_stride)                                                 \
  U(Uint32, rotary_width)                                                     \
  U(Uint32, max_rotary_length)                                                \
  U(Uint32, rotary_cache_batch_stride)                                        \
  U(Uint32, compress_ratio)                                                   \
  U(Uint32, capacity)                                                         \
  U(Uint32, past_sequence_length)                                             \
  U(Uint32, total_sequence_length)                                            \
  U(Uint32, block_topk)                                                       \
  U(Float32, epsilon)                                                         \
  U(Float32, scale)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_QSA_SELECT_PROGRAM_CONFIG

using SparseAttentionIndexerQsaSelectProgram = ConfiguredProgram<SparseAttentionIndexerQsaSelectProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_CONFIG(F)

struct SparseAttentionIndexerCsaCopyCompressedProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerCsaCopyCompressed";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                                                     \
  U(Uint32, head_size)                                                                 \
  U(Uint32, past_length)                                                               \
  U(Uint32, present_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_COMPRESSED_PROGRAM_CONFIG

using SparseAttentionIndexerCsaCopyCompressedProgram =
    ConfiguredProgram<SparseAttentionIndexerCsaCopyCompressedProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_CONFIG(F) F(bool, has_past_buffer_)

struct SparseAttentionIndexerCsaCompressProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past_buffer) {
    Config result{};
    result.has_past_buffer_ = bool{has_past_buffer};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerCsaCompress";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, work_items)                                                         \
  U(Uint32, sequence_length)                                                    \
  U(Uint32, head_size)                                                          \
  U(Uint32, rotary_width)                                                       \
  U(Uint32, max_rotary_length)                                                  \
  U(Uint32, rotary_cache_batch_stride)                                          \
  U(Uint32, compress_ratio)                                                     \
  U(Uint32, past_compressed_length)                                             \
  U(Uint32, present_compressed_length)                                          \
  U(Uint32, past_buffer_length)                                                 \
  U(Uint32, overlap_length)                                                     \
  U(Uint32, new_window_count)                                                   \
  U(Float32, epsilon)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COMPRESS_PROGRAM_CONFIG

using SparseAttentionIndexerCsaCompressProgram = ConfiguredProgram<SparseAttentionIndexerCsaCompressProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_CONFIG(F) \
  F(bool, has_past_buffer_)                                               \
  F(bool, has_current_)

struct SparseAttentionIndexerCsaCopyBufferProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past_buffer, bool has_current) {
    Config result{};
    result.has_past_buffer_ = bool{has_past_buffer};
    result.has_current_ = bool{has_current};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerCsaCopyBuffer";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, total)                                                                 \
  U(Uint32, sequence_length)                                                       \
  U(Uint32, head_size)                                                             \
  U(Uint32, past_buffer_length)                                                    \
  U(Uint32, present_buffer_length)                                                 \
  U(Uint32, present_buffer_start)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_COPY_BUFFER_PROGRAM_CONFIG

using SparseAttentionIndexerCsaCopyBufferProgram = ConfiguredProgram<SparseAttentionIndexerCsaCopyBufferProgramShader>;

#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG(F)

struct SparseAttentionIndexerCsaSelectProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "SparseAttentionIndexerCsaSelect";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);
#define WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, rows)                                                             \
  U(Uint32, sequence_length)                                                  \
  U(Uint32, num_heads)                                                        \
  U(Uint32, head_size)                                                        \
  U(Uint32, rotary_width)                                                     \
  U(Uint32, max_rotary_length)                                                \
  U(Uint32, rotary_cache_batch_stride)                                        \
  U(Uint32, compress_ratio)                                                   \
  U(Uint32, capacity)                                                         \
  U(Uint32, present_compressed_length)                                        \
  U(Float32, epsilon)                                                         \
  U(Float32, scale)                                                           \
  U(Float32, head_weight_scale)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPARSE_ATTENTION_INDEXER_CSA_SELECT_PROGRAM_SHADER_UNIFORMS
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
