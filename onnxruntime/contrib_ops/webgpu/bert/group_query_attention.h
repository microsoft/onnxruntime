// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class SplitPackedQKVProgram final : public Program<SplitPackedQKVProgram> {
 public:
  SplitPackedQKVProgram() : Program{"SplitPackedQKV"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kv_hidden_size", ProgramUniformVariableDataType::Uint32});
};

class SplitPackedQKVWithRotaryEmbeddingProgram final : public Program<SplitPackedQKVWithRotaryEmbeddingProgram> {
 public:
  SplitPackedQKVWithRotaryEmbeddingProgram(bool interleaved) : Program{"SplitPackedQKVWithRotaryEmbedding"}, interleaved_{interleaved} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override {
    // Inputs
    const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseUniform);
    const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
    const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform);
    const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);

    // Outputs
    const auto& query = sh.AddOutput("query", ShaderUsage::UseUniform);
    const auto& key = sh.AddOutput("key", ShaderUsage::UseUniform);
    const auto& val = sh.AddOutput("val", ShaderUsage::UseUniform);

    return WGSL_TEMPLATE_APPLY(sh, "bert/split_packed_qkv_with_rotary_embedding.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(interleaved, interleaved_),
                               WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                               WGSL_TEMPLATE_VARIABLE(key, key),
                               WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                               WGSL_TEMPLATE_VARIABLE(query, query),
                               WGSL_TEMPLATE_VARIABLE(seqlens, seqlens),
                               WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache),
                               WGSL_TEMPLATE_VARIABLE(val, val));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"half_rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"first_prompt_flag", ProgramUniformVariableDataType::Uint32},
      {"subsequent_prompt_flag", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool interleaved_;
};

class GroupQueryAttention final : public WebGpuKernel {
 public:
  GroupQueryAttention(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
    kv_num_heads_ = static_cast<int>(kv_num_heads);

    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);

    do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

    use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;

    local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  }

  int num_heads_;     // number of attention heads of Q
  int kv_num_heads_;  // number of attention heads of K or V
  float scale_;       // the scaling factor applied before softmax
  float softcap_;
  bool do_rotary_;  // whether or not to use rotary embeddings
  bool rotary_interleaved_;
  int local_window_size_;

  bool use_smooth_softmax_;
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
