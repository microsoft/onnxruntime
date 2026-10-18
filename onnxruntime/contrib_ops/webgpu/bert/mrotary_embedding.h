// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "contrib_ops/cpu/bert/mrotary_embedding_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class MRotaryEmbeddingProgram final : public Program<MRotaryEmbeddingProgram> {
 public:
  MRotaryEmbeddingProgram(bool interleaved, bool transposed,
                          mrotary_embedding_helper::MRopeLayout mrope_layout)
      : Program{"MRotaryEmbedding"},
        interleaved_{interleaved},
        transposed_{transposed},
        mrope_layout_{mrope_layout} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

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

 private:
  const bool interleaved_;
  const bool transposed_;
  const mrotary_embedding_helper::MRopeLayout mrope_layout_;
};

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
