// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class CopyKVCacheProgram final : public Program<CopyKVCacheProgram> {
 public:
  CopyKVCacheProgram(const std::string& kernel_name, int components, bool has_past)
      : Program{kernel_name}, components_(components), has_past_(has_past) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"past_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"vectorized_head_size", ProgramUniformVariableDataType::Uint32});

 private:
  int components_;
  bool has_past_;
};

class FlashAttentionProgram final : public Program<FlashAttentionProgram> {
 public:
  FlashAttentionProgram(const std::string& kernel_name,
                        bool has_attention_bias,
                        int qkv_head_size,
                        int qkv_num_heads)
      : Program{kernel_name},
        has_attention_bias_(has_attention_bias),
        qkv_head_size_(qkv_head_size),
        qkv_num_heads_(qkv_num_heads) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32});

 private:
  bool has_attention_bias_;
  int qkv_head_size_;
  int qkv_num_heads_;
};

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context);

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context);
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
