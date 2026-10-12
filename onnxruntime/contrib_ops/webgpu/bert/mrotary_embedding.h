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
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_CONFIG);
  static Config MakeConfig(bool interleaved, bool transposed, mrotary_embedding_helper::MRopeLayout mrope_layout) {
    Config result{};
    result.interleaved_ = bool{interleaved};
    result.transposed_ = bool{transposed};
    result.mrope_layout_ = mrotary_embedding_helper::MRopeLayout{mrope_layout};

    return result;
  }
  static constexpr std::string_view name = "MRotaryEmbedding";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader);

#define WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_SHADER_UNIFORMS(U) \
  U(Float32, scale)                                          \
  U(Uint32, output_size)                                     \
  U(Uint32, batch_size)                                      \
  U(Uint32, sequence_length)                                 \
  U(Uint32, num_heads)                                       \
  U(Uint32, head_size)                                       \
  U(Uint32, rotary_embedding_dim)                            \
  U(Uint32, max_sequence_length)                             \
  U(Uint32, mrope_section)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_M_ROTARY_EMBEDDING_PROGRAM_SHADER_UNIFORMS
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
