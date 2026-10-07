// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "contrib_ops/cpu/bert/mrotary_embedding_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_CONFIG(F) \
  F(bool, interleaved_)                             \
  F(bool, transposed_)                              \
  F(mrotary_embedding_helper::MRopeLayout, mrope_layout_)

struct MRotaryEmbeddingProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_CONFIG);
    Config(bool interleaved, bool transposed, mrotary_embedding_helper::MRopeLayout mrope_layout)
        : interleaved_{interleaved}, transposed_{transposed}, mrope_layout_{mrope_layout} {}
  };
  static constexpr std::string_view name = "MRotaryEmbedding";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"scale", ProgramUniformVariableDataType::Float32},
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"rotary_embedding_dim", ProgramUniformVariableDataType::Uint32},
      {"max_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"mrope_section", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_CONFIG

using MRotaryEmbeddingProgram = ConfiguredProgram<MRotaryEmbeddingProgramShader>;

class MRotaryEmbedding final : public WebGpuKernel {
 public:
  explicit MRotaryEmbedding(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  float scale_;
  int num_heads_;
  int rotary_embedding_dim_;
  bool interleaved_;
  bool is_packed_batching_;
  int64_t mrope_layout_;
  std::vector<int64_t> mrope_section_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
